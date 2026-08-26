# MMLU full 5-shot — 2026-08-27

DeepSeek V4 Flash JA REAP K216を標準MMLU全57分野で評価した結果です。以前の固定200問0-shot subsetとは別の、全14,042問・5-shot・log-likelihood評価です。

## 結果

- **総合: 12,081 / 14,042 = 86.03%**
- Wilson 95%信頼区間: **85.45%–86.60%**

| カテゴリ | 正答数 | Accuracy |
|---|---:|---:|
| Social Sciences | 2825/3077 | 91.81% |
| Stem | 2783/3153 | 88.27% |
| Other | 2701/3107 | 86.93% |
| Humanities | 3772/4705 | 80.17% |

## 評価条件

```text
benchmark: MMLU full (57 subjects, 14,042 questions)
num_fewshot: 5
batch_size: 1
num_concurrent: 1
backend: lm-eval local-completions /v1/completions
tokenizer: deepseek-ai/DeepSeek-V4-Flash-0731
tokenized_requests: false
lm-eval: 0.4.9.1
seeds: random=0, numpy=1234, torch=1234, fewshot=1234
model: DeepSeek-V4-Flash-0731-JA-REAP-K216 EXL3 3.0 bpw TP1
draft: K64 DSpark, K5 probabilistic
context: 256,000
GPU memory utilization: 0.945
```

## 57分野

| 分野 | 正答数 | Accuracy |
|---|---:|---:|
| `abstract_algebra` | 80/100 | 80.00% |
| `anatomy` | 108/135 | 80.00% |
| `astronomy` | 143/152 | 94.08% |
| `business_ethics` | 87/100 | 87.00% |
| `clinical_knowledge` | 225/265 | 84.91% |
| `college_biology` | 136/144 | 94.44% |
| `college_chemistry` | 66/100 | 66.00% |
| `college_computer_science` | 93/100 | 93.00% |
| `college_mathematics` | 82/100 | 82.00% |
| `college_medicine` | 147/173 | 84.97% |
| `college_physics` | 96/102 | 94.12% |
| `computer_security` | 84/100 | 84.00% |
| `conceptual_physics` | 225/235 | 95.74% |
| `econometrics` | 91/114 | 79.82% |
| `electrical_engineering` | 126/145 | 86.90% |
| `elementary_mathematics` | 359/378 | 94.97% |
| `formal_logic` | 94/126 | 74.60% |
| `global_facts` | 65/100 | 65.00% |
| `high_school_biology` | 291/310 | 93.87% |
| `high_school_chemistry` | 173/203 | 85.22% |
| `high_school_computer_science` | 95/100 | 95.00% |
| `high_school_european_history` | 144/165 | 87.27% |
| `high_school_geography` | 187/198 | 94.44% |
| `high_school_government_and_politics` | 189/193 | 97.93% |
| `high_school_macroeconomics` | 364/390 | 93.33% |
| `high_school_mathematics` | 211/270 | 78.15% |
| `high_school_microeconomics` | 229/238 | 96.22% |
| `high_school_physics` | 129/151 | 85.43% |
| `high_school_psychology` | 521/545 | 95.60% |
| `high_school_statistics` | 197/216 | 91.20% |
| `high_school_us_history` | 189/204 | 92.65% |
| `high_school_world_history` | 220/237 | 92.83% |
| `human_aging` | 189/223 | 84.75% |
| `human_sexuality` | 116/131 | 88.55% |
| `international_law` | 110/121 | 90.91% |
| `jurisprudence` | 97/108 | 89.81% |
| `logical_fallacies` | 143/163 | 87.73% |
| `machine_learning` | 89/112 | 79.46% |
| `management` | 92/103 | 89.32% |
| `marketing` | 226/234 | 96.58% |
| `medical_genetics` | 96/100 | 96.00% |
| `miscellaneous` | 720/783 | 91.95% |
| `moral_disputes` | 302/346 | 87.28% |
| `moral_scenarios` | 688/895 | 76.87% |
| `nutrition` | 279/306 | 91.18% |
| `philosophy` | 276/311 | 88.75% |
| `prehistory` | 270/324 | 83.33% |
| `professional_accounting` | 238/282 | 84.40% |
| `professional_law` | 1101/1534 | 71.77% |
| `professional_medicine` | 244/272 | 89.71% |
| `professional_psychology` | 547/612 | 89.38% |
| `public_relations` | 85/110 | 77.27% |
| `security_studies` | 214/245 | 87.35% |
| `sociology` | 191/201 | 95.02% |
| `us_foreign_policy` | 91/100 | 91.00% |
| `virology` | 93/166 | 56.02% |
| `world_religions` | 138/171 | 80.70% |

## 再現性と成果物

- 57個のper-subject sample JSONL、計14,042レコードをローカルで照合済み。
- request cacheは56,060個の一意キー。問題数×4との差は同一リクエストの重複排除による。
- 生サンプル、SQLite cache、ローカルパス、環境dumpはサイズ・プライバシーのためGitへ含めない。
- 集計値と全57分野の機械可読データは[`mmlu-full-5shot-20260827.json`](mmlu-full-5shot-20260827.json)に保存。

## モデルとクレジット

- Model: [Laplace1313/DeepSeek-V4-Flash-0731-JA-REAP-K216-EXL3-3bpw-DGX-Spark](https://huggingface.co/Laplace1313/DeepSeek-V4-Flash-0731-JA-REAP-K216-EXL3-3bpw-DGX-Spark)
- Base model: [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- ライセンスと上流クレジットはリポジトリの[`LICENSES/`](../LICENSES/)および[`NOTICE.md`](../NOTICE.md)を参照。
