# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculator: semi-autoregressive parallel drafting.

DSpark drafts a block of ``num_speculative_tokens`` tokens in one parallel pass
(reusing the DFlash machinery: context-KV precompute + a query-block forward),
then injects intra-block dependency with a lightweight sequential Markov head.

Differences from DFlash:
  * Anchor-as-first-prediction: each request emits exactly ``N =
    num_speculative_tokens`` query tokens (anchor + N-1 noise), NOT ``1 + N``.
    Every query position is a prediction (the anchor predicts the first draft
    token), so we sample at all N positions and ``sample_pos = query_pos + 1``
    (standard next-token), whereas DFlash's masks sit AT the predicted position.
    This is the ``sample_from_anchor`` path in the shared prepare-inputs kernel.
    Speculators-format checkpoints instead use the DFlash ``1 + N`` fill-in
    layout (anchor is the bonus token).
  * Sequential Markov sampling: instead of DFlash's single parallel sample, we
    sample left-to-right, adding a prefix-dependent Markov bias derived from the
    previously sampled token at each step.

CUDA graphs (FULL, mirroring DFlash) cover the whole draft step: the parallel
backbone forward AND the sequential Markov sampling.
"""

import json
import os
from typing import Any

import torch
from safetensors import safe_open

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.triton_utils import triton
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.capacity import (
    build_sps_table,
    compute_draft_token_capacity_from_confidence,
)
from vllm.v1.worker.gpu.spec_decode.dspark.online_sts import DSparkOnlineSTS
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model


class DSparkSpeculator(DFlashSpeculator):
    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        # Anchor-as-first (N slots) unless the checkpoint uses the 1+N fill-in
        # block, where the anchor is a separate bonus token.
        self.sample_from_anchor = not getattr(
            self.draft_model_config.hf_config, "dspark_bonus_anchor", False
        )
        if self.sample_from_anchor:
            self.num_query_per_req = self.num_speculative_steps
        else:
            self.num_query_per_req = 1 + self.num_speculative_steps

        # DSpark consumes mean-pooled target aux hidden states at the target
        # layers, combined to hidden_size via main_proj. Store that combined
        # main_x (hidden_size wide). DSpark does not use the same pre-allocated buffer
        # that DeepSeek-V4's MTP uses.
        draft_hidden = self.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            self.max_num_tokens, draft_hidden, dtype=self.dtype, device=device
        )

        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )

        # Reduced-vocab probabilistic drafting only; set in load_draft_model.
        self._d2t_scatter_index: torch.Tensor | None = None
        self._draft_scatter_buf: torch.Tensor | None = None
        draft_topk = int(os.environ.get("VLLM_DSPARK_DRAFT_TOPK", "0") or "0")
        if not 0 <= draft_topk <= self.vocab_size:
            raise ValueError("DSpark draft top-k is outside vocabulary")
        if (
            draft_topk
            and self.vllm_config.parallel_config.tensor_parallel_size != 1
        ):
            raise ValueError("DSpark draft top-k requires tensor parallel size 1")
        self._draft_topk: int | None = draft_topk or None

        self.draft_token_confidence_logits = torch.empty(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.float32,
            device=device,
        )
        self.draft_token_survival_probs = torch.empty_like(
            self.draft_token_confidence_logits
        )
        self.draft_token_capacity = torch.full(
            (self.max_num_reqs,),
            self.num_speculative_steps,
            dtype=torch.int32,
            device=device,
        )
        self.capacity_activation_batch_size = (
            envs.VLLM_DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE
        )
        if self.capacity_activation_batch_size < 0:
            raise ValueError(
                "VLLM_DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE must be >= 0, got "
                f"{self.capacity_activation_batch_size}."
            )
        self._runtime_num_reqs_for_capacity = torch.zeros(
            (1,),
            dtype=torch.int32,
            device=device,
        )
        self.draft_token_valid_lengths = torch.empty(
            (self.max_num_reqs,),
            dtype=torch.int32,
            device=device,
        )
        self._last_num_speculative_steps = self.num_speculative_steps
        self._last_proposal_confidence_valid = False
        self._training_hidden_path = os.environ.get(
            "VLLM_DSPARK_TRAINING_HIDDEN_PATH"
        )
        self._training_hidden_meta_path = os.environ.get(
            "VLLM_DSPARK_TRAINING_HIDDEN_META_PATH"
        )
        self._training_round_limit = int(
            os.environ.get("VLLM_DSPARK_TRAINING_ROUND_LIMIT", "0") or "0"
        )
        self._training_round_index = 0
        self._hidden_adapter_w1: torch.Tensor | None = None
        self._hidden_adapter_w2: torch.Tensor | None = None
        self._position_adapter_w1: torch.Tensor | None = None
        self._position_adapter_w2: torch.Tensor | None = None
        self._position_adapter_scale = float(
            os.environ.get("VLLM_DSPARK_POSITION_ADAPTER_SCALE", "1.0") or "1.0"
        )
        self._position_adapter_topk = int(
            os.environ.get("VLLM_DSPARK_POSITION_ADAPTER_TOPK", "0") or "0"
        )
        if not 0 <= self._position_adapter_topk <= self.vocab_size:
            raise ValueError("DSpark position adapter top-k is outside vocabulary")
        self._position_adapter_activation = os.environ.get(
            "VLLM_DSPARK_POSITION_ADAPTER_ACTIVATION", "none"
        ).lower()
        if self._position_adapter_activation not in {"none", "gelu"}:
            raise ValueError("unsupported DSpark position adapter activation")
        position_adapter_path = os.environ.get("VLLM_DSPARK_POSITION_ADAPTER_PATH")
        if position_adapter_path:
            with safe_open(position_adapter_path, framework="pt", device="cpu") as handle:
                position_w1 = handle.get_tensor("position_adapter_w1")
                position_w2 = handle.get_tensor("position_adapter_w2")
            expected_positions = self.num_speculative_steps
            if position_w1.shape[0] != expected_positions:
                raise ValueError("DSpark position adapter block size mismatch")
            if position_w1.shape[1] != draft_hidden:
                raise ValueError("DSpark position adapter hidden size mismatch")
            if position_w1.shape[2] != position_w2.shape[2]:
                raise ValueError("DSpark position adapter ranks do not match")
            if position_w2.shape[:2] != (expected_positions, self.vocab_size):
                raise ValueError("DSpark position adapter vocab size mismatch")
            self._position_adapter_w1 = position_w1.to(device=device, dtype=self.dtype)
            self._position_adapter_w2 = position_w2.to(device=device, dtype=self.dtype)
        self._hidden_transform_a: torch.Tensor | None = None
        self._hidden_transform_b: torch.Tensor | None = None
        self._shifted_hidden_transform_a: torch.Tensor | None = None
        self._shifted_hidden_transform_b: torch.Tensor | None = None
        self._shifted_hidden_transform_scale = float(
            os.environ.get(
                "VLLM_DSPARK_SHIFTED_HIDDEN_TRANSFORM_SCALE", "1.0"
            )
            or "1.0"
        )
        shifted_hidden_path = os.environ.get(
            "VLLM_DSPARK_SHIFTED_HIDDEN_TRANSFORM_PATH"
        )
        if shifted_hidden_path:
            with safe_open(shifted_hidden_path, framework="pt", device="cpu") as handle:
                shifted_a = handle.get_tensor("shifted_hidden_transform_a")
                shifted_b = handle.get_tensor("shifted_hidden_transform_b")
            expected_positions = self.num_speculative_steps
            if shifted_a.shape[:2] != (expected_positions, draft_hidden):
                raise ValueError("DSpark shifted hidden transform input shape mismatch")
            if shifted_b.shape[0] != expected_positions:
                raise ValueError("DSpark shifted hidden transform block size mismatch")
            if shifted_a.shape[2] != shifted_b.shape[1]:
                raise ValueError("DSpark shifted hidden transform ranks do not match")
            if shifted_b.shape[2] != draft_hidden:
                raise ValueError("DSpark shifted hidden transform output size mismatch")
            self._shifted_hidden_transform_a = shifted_a.to(
                device=device, dtype=self.dtype
            )
            self._shifted_hidden_transform_b = shifted_b.to(
                device=device, dtype=self.dtype
            )
        self._hidden_transform_scale = float(
            os.environ.get("VLLM_DSPARK_HIDDEN_TRANSFORM_SCALE", "1.0") or "1.0"
        )
        hidden_transform_path = os.environ.get("VLLM_DSPARK_HIDDEN_TRANSFORM_PATH")
        if hidden_transform_path:
            with safe_open(hidden_transform_path, framework="pt", device="cpu") as handle:
                transform_a = handle.get_tensor("hidden_transform_a")
                transform_b = handle.get_tensor("hidden_transform_b")
            if transform_a.shape[0] != draft_hidden:
                raise ValueError("DSpark hidden transform input size mismatch")
            if transform_a.shape[1] != transform_b.shape[0]:
                raise ValueError("DSpark hidden transform ranks do not match")
            if transform_b.shape[1] != draft_hidden:
                raise ValueError("DSpark hidden transform output size mismatch")
            self._hidden_transform_a = transform_a.to(device=device, dtype=self.dtype)
            self._hidden_transform_b = transform_b.to(device=device, dtype=self.dtype)
        self._hidden_adapter_scale = float(
            os.environ.get("VLLM_DSPARK_HIDDEN_ADAPTER_SCALE", "1.0") or "1.0"
        )
        hidden_adapter_path = os.environ.get("VLLM_DSPARK_HIDDEN_ADAPTER_PATH")
        if hidden_adapter_path:
            with safe_open(hidden_adapter_path, framework="pt", device="cpu") as handle:
                adapter_w1 = handle.get_tensor("hidden_adapter_w1")
                adapter_w2 = handle.get_tensor("hidden_adapter_w2")
            if adapter_w1.ndim != 2 or adapter_w2.ndim != 2:
                raise ValueError("DSpark hidden adapter tensors must be rank-2")
            if adapter_w1.shape[0] != draft_hidden:
                raise ValueError(
                    f"hidden adapter input {adapter_w1.shape[0]} != {draft_hidden}"
                )
            if adapter_w1.shape[1] != adapter_w2.shape[1]:
                raise ValueError("DSpark hidden adapter ranks do not match")
            if adapter_w2.shape[0] != self.vocab_size:
                raise ValueError(
                    f"hidden adapter vocab {adapter_w2.shape[0]} != {self.vocab_size}"
                )
            self._hidden_adapter_w1 = adapter_w1.to(device=device, dtype=self.dtype)
            self._hidden_adapter_w2 = adapter_w2.to(device=device, dtype=self.dtype)
        self.min_survival_probability = (
            self.speculative_config.dspark_confidence_threshold
        )
        self.capacity_budget_frac = self.speculative_config.dspark_budget_frac
        self.confidence_temperature = (
            self.speculative_config.dspark_confidence_temperature
        )
        sps_curve = self.speculative_config.dspark_sps_curve
        self.sps_table: torch.Tensor | None = None
        self.wants_auto_sps_curve = sps_curve == "auto"
        if sps_curve is not None:
            # Sized for the pow2-padded request count the allocator kernel
            # can index under CUDA graph capture.
            padded_reqs = triton.next_power_of_2(max(self.max_num_reqs, 1))
            max_batch_tokens = padded_reqs * (1 + self.num_speculative_steps)
            if self.wants_auto_sps_curve:
                # Flat placeholder (theta argmax verifies everything) until
                # the post-capture profiling refreshes the contents in place;
                # the captured allocator kernel bakes this buffer's address.
                self.sps_table = torch.ones(
                    max_batch_tokens + 1, dtype=torch.float32, device=device
                )
            else:
                assert isinstance(sps_curve, list)
                self.sps_table = build_sps_table(
                    sps_curve,
                    max_batch_tokens,
                    device,
                )
        self.use_draft_token_capacity = (
            self.min_survival_probability > 0.0
            or self.capacity_budget_frac < 1.0
            or self.sps_table is not None
        )
        self.online_sts: DSparkOnlineSTS | None = None
        if self.use_draft_token_capacity and self.speculative_config.dspark_online_sts:
            self.online_sts = DSparkOnlineSTS(
                self.max_num_reqs, self.num_speculative_steps, device
            )
            # Calibrated survival buffer consumed by the capacity kernels
            # inside the captured draft graph.
            self.calibrated_confidence_logits = torch.zeros_like(
                self.draft_token_confidence_logits
            )

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        confidence_head = getattr(
            getattr(model, "model", None), "confidence_head", None
        )
        if self.use_draft_token_capacity and (
            getattr(model, "compute_confidence", None) is None
            or confidence_head is None
        ):
            raise ValueError(
                "DSpark draft-token capacity requires a draft model with a "
                f"confidence head; {type(model).__name__} does not implement "
                "compute_confidence."
            )
        # Reduced draft vocab: probabilistic rejection sampling indexes draft
        # logits by target id, so precompute the draft->target column map and a
        # scratch buffer to scatter logits into target vocab before sampling.
        d2t = getattr(model, "draft_id_to_target_id", None)
        if self.draft_logits is not None and d2t is not None:
            self._d2t_scatter_index = (
                torch.arange(d2t.shape[0], device=d2t.device) + d2t
            )
            # -inf once; the per-step scatter overwrites the draft->target
            # columns. Kept separate from draft_logits to avoid aliasing.
            self._draft_scatter_buf = torch.full(
                (self.max_num_reqs, self.vocab_size),
                float("-inf"),
                dtype=self.draft_logits.dtype,
                device=self.device,
            )
        return model

    def _record_training_hidden(
        self,
        *,
        sample_hidden: torch.Tensor,
        num_reqs: int,
        num_speculative_steps: int,
    ) -> None:
        hidden_path = self._training_hidden_path
        meta_path = self._training_hidden_meta_path
        limit = self._training_round_limit
        if (
            not hidden_path
            or not meta_path
            or num_reqs != 1
            or (limit > 0 and self._training_round_index >= limit)
        ):
            return
        hidden = (
            sample_hidden[:1, :num_speculative_steps]
            .detach()
            .to(device="cpu", dtype=torch.bfloat16)
            .contiguous()
        )
        raw = hidden.view(torch.uint8).numpy().tobytes()
        os.makedirs(os.path.dirname(hidden_path) or ".", exist_ok=True)
        offset = os.path.getsize(hidden_path) if os.path.exists(hidden_path) else 0
        with open(hidden_path, "ab") as handle:
            handle.write(raw)
            handle.flush()
        record = {
            "round_index": self._training_round_index,
            "offset": offset,
            "length": len(raw),
            "shape": list(hidden.shape),
            "dtype": "BF16",
        }
        with open(meta_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            handle.flush()
        self._training_round_index += 1

    def _sample_sequential_topk(
        self,
        *,
        markov_embed: torch.Tensor,
        logits: torch.Tensor,
        values: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the sequential Markov head only to selected base candidates."""
        return self.model.apply_markov_bias_gathered(
            markov_embed,
            logits,
            values,
            indices,
        )

    def _sample_sequential(
        self,
        num_reqs: int,
        head_hidden: torch.Tensor,
        num_speculative_steps: int,
        num_query_per_req: int,
        is_profile: bool = False,
        use_capacity: bool = True,
    ) -> None:
        # Sequential Markov sampling over the backbone's output hidden states.
        n_spec = num_speculative_steps
        num_sample = num_reqs * n_spec
        # Per-(req, position) head hidden, ordered (req, step).
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        sample_hidden = sample_hidden.view(num_reqs, n_spec, -1)
        # Draft-vocab logits; sampled ids are remapped to target vocab below.
        logit_hidden = sample_hidden
        if self._shifted_hidden_transform_a is not None:
            assert self._shifted_hidden_transform_b is not None
            shifted_a = self._shifted_hidden_transform_a[1:n_spec]
            shifted_b = self._shifted_hidden_transform_b[1:n_spec]
            shifted_source = sample_hidden[:, 1:].to(shifted_a.dtype)
            shifted_latent = torch.einsum(
                "bph,phr->bpr", shifted_source, shifted_a
            )
            shifted_delta = torch.einsum(
                "bpr,prh->bph", shifted_latent, shifted_b
            )
            shifted_prefix = logit_hidden[:, :-1] + shifted_delta.to(
                logit_hidden.dtype
            ) * float(self._shifted_hidden_transform_scale)
            logit_hidden = torch.cat(
                (shifted_prefix, logit_hidden[:, -1:]), dim=1
            )
        if self._hidden_transform_a is not None:
            assert self._hidden_transform_b is not None
            transform_input = sample_hidden.to(self._hidden_transform_a.dtype)
            transform_delta = torch.matmul(
                torch.matmul(transform_input, self._hidden_transform_a),
                self._hidden_transform_b,
            )
            transform_position_mask = (
                self._step_cols[:n_spec]
                .gt(0)
                .view(1, n_spec, 1)
                .to(transform_delta.dtype)
            )
            transform_delta = transform_delta * transform_position_mask
            logit_hidden = sample_hidden + transform_delta.to(sample_hidden.dtype) * float(
                self._hidden_transform_scale
            )
        base_logits = self.model.compute_draft_logits(
            logit_hidden.reshape(num_sample, -1)
        )
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)
        if self._position_adapter_w1 is not None:
            assert self._position_adapter_w2 is not None
            position_w1 = self._position_adapter_w1[:n_spec]
            position_w2 = self._position_adapter_w2[:n_spec]
            position_latent = torch.einsum(
                "bph,phr->bpr",
                sample_hidden.to(position_w1.dtype),
                position_w1,
            )
            if self._position_adapter_activation == "gelu":
                position_latent = torch.nn.functional.gelu(position_latent)
            # Verifier/draft alignment is shifted by one: adapter position 1
            # was trained from verifier logit row 0, position 2 from row 1,
            # and so on. Position 0 has no trainable verifier row, while the
            # final base-logit row has no matching suffix adapter position.
            if self._position_adapter_topk > 0:
                suffix_base = base_logits[:, :-1]
                position_topk_ids = suffix_base.topk(
                    self._position_adapter_topk, dim=-1
                ).indices
                suffix_w2 = position_w2[1:n_spec]
                expanded_w2 = suffix_w2.unsqueeze(0).expand(
                    num_reqs, -1, -1, -1
                )
                selected_w2 = torch.gather(
                    expanded_w2,
                    2,
                    position_topk_ids.unsqueeze(-1).expand(
                        -1, -1, -1, suffix_w2.shape[-1]
                    ),
                )
                selected_bias = torch.einsum(
                    "bpr,bpkr->bpk", position_latent[:, 1:], selected_w2
                ).to(base_logits.dtype) * float(self._position_adapter_scale)
                suffix_logits = suffix_base.scatter_add(
                    -1, position_topk_ids, selected_bias
                )
            else:
                position_bias = torch.einsum(
                    "bpr,pvr->bpv", position_latent, position_w2
                )
                suffix_logits = base_logits[:, :-1] + position_bias[:, 1:].to(
                    base_logits.dtype
                ) * float(self._position_adapter_scale)
            base_logits = torch.cat((suffix_logits, base_logits[:, -1:]), dim=1)
        if self._hidden_adapter_w1 is not None:
            assert self._hidden_adapter_w2 is not None
            adapter_latent = torch.matmul(
                sample_hidden.to(self._hidden_adapter_w1.dtype),
                self._hidden_adapter_w1,
            )
            adapter_bias = torch.matmul(
                adapter_latent, self._hidden_adapter_w2.transpose(0, 1)
            )
            base_logits = base_logits + adapter_bias.to(base_logits.dtype) * float(
                self._hidden_adapter_scale
            )

        if self._draft_topk is not None:
            if self._draft_topk > base_logits.shape[-1]:
                raise ValueError("DSpark draft top-k exceeds the draft vocabulary")
            base_values, draft_indices = base_logits.topk(self._draft_topk, dim=-1)
            # Reuse the dense backbone output as the normal sampler input. Only
            # selected rows are restored after the gathered Markov projection.
            base_logits.fill_(float("-inf"))
        else:
            base_values = None
            draft_indices = None

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        confidence_logits = self.draft_token_confidence_logits[:num_reqs, :n_spec]
        min_survival_probability = self.min_survival_probability
        use_confidence_capacity = self.use_draft_token_capacity and use_capacity

        # Anchor (bonus) token per request = the input id at query offset 0,
        # laid out as one row per request in the draft query block.
        prev = self.input_buffers.input_ids[
            : num_reqs * num_query_per_req : num_query_per_req
        ]
        valid_prefix = torch.ones(num_reqs, dtype=torch.bool, device=self.device)
        valid_lengths = self.draft_token_valid_lengths[:num_reqs]
        valid_lengths.zero_()

        for i in range(n_spec):
            # Sequential stage: Markov bias from the previously sampled token.
            markov_embed = self.model.markov_embed(prev)
            if use_confidence_capacity:
                confidence_i = self.model.compute_confidence(
                    sample_hidden[:, i], markov_embed
                )
                if confidence_i is None:
                    raise RuntimeError(
                        "DSpark draft-token capacity requires loaded "
                        "confidence-head weights."
                    )
                confidence_logits[:, i] = confidence_i
            if self._draft_topk is not None:
                assert base_values is not None and draft_indices is not None
                logits_i = self._sample_sequential_topk(
                    markov_embed=markov_embed,
                    logits=base_logits[:, i],
                    values=base_values[:, i],
                    indices=draft_indices[:, i],
                )
            else:
                bias = self.model.markov_bias(markov_embed)
                logits_i = base_logits[:, i] + bias
            if self.draft_logits is not None:
                # Probabilistic: sample in target vocab (a reduced draft vocab is
                # scattered into its target columns; full vocab is already there).
                if self._d2t_scatter_index is not None:
                    assert self._draft_scatter_buf is not None
                    buf = self._draft_scatter_buf[:num_reqs]
                    buf.index_copy_(1, self._d2t_scatter_index, logits_i.to(buf.dtype))
                    logits_i = buf
                # sample_pos is the predicted token's position Q. The shared
                # sampler adds one before salting, so Q-2 produces a unique
                # draft key for each predicted position.
                draft_sampled_i = self._sample_probabilistic_draft(
                    logits=logits_i,
                    positions=sample_pos[:, i] - 2,
                    idx_mapping=idx_map[:, i],
                    temperature=self.temperature,
                    seeds=self.seeds,
                    draft_step=self._step_cols[i],
                    draft_logits=self.draft_logits,
                )
            else:
                draft_sampled_i = self.model.map_draft_to_target(
                    logits_i.argmax(dim=-1)
                )
            valid_prefix.logical_and_(
                (draft_sampled_i >= 0) & (draft_sampled_i < self.vocab_size)
            )
            draft_sampled_i = torch.where(
                valid_prefix, draft_sampled_i, torch.zeros_like(draft_sampled_i)
            )
            valid_lengths.add_(valid_prefix.to(torch.int32))
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if not is_profile and not torch.cuda.is_current_stream_capturing():
            self._record_training_hidden(
                sample_hidden=sample_hidden,
                num_reqs=num_reqs,
                num_speculative_steps=n_spec,
            )

        if use_confidence_capacity and not is_profile:
            capacity_confidence = self.draft_token_confidence_logits
            capacity_temperature = self.confidence_temperature
            if self.online_sts is not None:
                self.online_sts.calibrate(
                    confidence_logits,
                    out=self.calibrated_confidence_logits[:num_reqs, :n_spec],
                )
                capacity_confidence = self.calibrated_confidence_logits
                capacity_temperature = 1.0
            compute_draft_token_capacity_from_confidence(
                capacity_confidence,
                self.draft_token_capacity,
                min_survival_probability,
                num_reqs,
                n_spec,
                self._runtime_num_reqs_for_capacity,
                self.draft_token_survival_probs,
                self.capacity_budget_frac,
                sps_table=self.sps_table,
                confidence_temperature=capacity_temperature,
            )
        else:
            self.draft_token_capacity[:num_reqs].fill_(n_spec)
        torch.minimum(
            self.draft_token_capacity[:num_reqs],
            valid_lengths,
            out=self.draft_token_capacity[:num_reqs],
        )

    def set_sps_curve(self, sps_curve: list[tuple[int, float]]) -> None:
        """Refresh the SPS lookup table in place (its address is baked into
        the captured allocator kernel)."""
        assert self.sps_table is not None
        dense = build_sps_table(
            sps_curve, self.sps_table.shape[0] - 1, self.sps_table.device
        )
        self.sps_table.copy_(dense)

    def compute_capacities(self, input_batch: InputBatch) -> torch.Tensor | None:
        if not self.use_draft_token_capacity:
            return None
        num_reqs = input_batch.num_reqs
        if self.online_sts is not None:
            # Join key for verification outcomes arriving next step. Staged
            # eagerly (not in the captured graph): a padded replay would
            # index_put through stale padding-row ids, and -1 sentinels wrap
            # to the last row, so neither is safe for a scatter by slot.
            n_spec = self._last_num_speculative_steps
            self.online_sts.stage_proposal(
                self.sample_idx_mapping[: num_reqs * n_spec : n_spec],
                self.draft_token_confidence_logits[:num_reqs, :n_spec],
                valid=self._last_proposal_confidence_valid,
            )
        return self.draft_token_capacity[:num_reqs]

    def warmup_capacity_kernels(self) -> None:
        self._warmup_prepare_inputs_kernel()
        if not self.use_draft_token_capacity:
            return

        self.draft_token_confidence_logits.zero_()
        sizes = {self.max_num_reqs}
        num_reqs = 1
        while num_reqs < self.max_num_reqs:
            sizes.add(num_reqs)
            num_reqs *= 2
        for num_reqs in sorted(sizes):
            self._runtime_num_reqs_for_capacity.fill_(num_reqs)
            compute_draft_token_capacity_from_confidence(
                self.draft_token_confidence_logits,
                self.draft_token_capacity,
                self.min_survival_probability,
                num_reqs,
                self.num_speculative_steps,
                self._runtime_num_reqs_for_capacity,
                self.draft_token_survival_probs,
                self.capacity_budget_frac,
                sps_table=self.sps_table,
                confidence_temperature=self.confidence_temperature,
            )

    def propose(
        self,
        input_batch: InputBatch,
        *args,
        num_speculative_tokens: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.use_draft_token_capacity:
            self._runtime_num_reqs_for_capacity.fill_(input_batch.num_reqs)
        self._last_proposal_confidence_valid = bool(
            self.use_draft_token_capacity
            and not kwargs.get("is_profile", False)
            and not kwargs.get("dummy_run", False)
            and not self._has_unaligned_cached_prefix(input_batch)
            and (
                self.capacity_activation_batch_size <= 0
                or input_batch.num_reqs >= self.capacity_activation_batch_size
            )
        )
        self._last_num_speculative_steps = (
            num_speculative_tokens
            if self.dynamic_physical_depth and num_speculative_tokens is not None
            else self.num_speculative_steps
        )
        return super().propose(
            input_batch,
            *args,
            num_speculative_tokens=num_speculative_tokens,
            **kwargs,
        )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        is_profile: bool = False,
        num_query_per_req: int | None = None,
    ) -> None:
        if num_query_per_req is None:
            num_query_per_req = self.num_query_per_req
        num_speculative_steps = self._speculative_steps_for_query_len(num_query_per_req)
        # Full draft step (captured under CUDA graph): parallel backbone forward
        # then sequential Markov sampling over its hidden state outputs.
        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        self._sample_sequential(
            num_reqs,
            head_hidden,
            num_speculative_steps,
            num_query_per_req,
            is_profile=is_profile,
            use_capacity=(
                self.capacity_activation_batch_size <= 0
                or num_reqs >= self.capacity_activation_batch_size
            ),
        )
