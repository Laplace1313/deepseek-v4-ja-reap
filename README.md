# DeepSeek V4 Flash JA REAP — DGX Spark向け

DeepSeek V4 Flash-0731を、日本語、ツール呼び出し、コーディング向けに調整したDGX Spark 1台用モデルです。

[Hugging Faceからモデルをダウンロード](https://huggingface.co/Laplace1313/DeepSeek-V4-Flash-0731-JA-REAP-K216-EXL3-3bpw-DGX-Spark) · [ランタイムのソース](runtime/) · [ライセンスとクレジット](NOTICE.md)

## 概要

| 項目 | 内容 |
|---|---|
| ベースモデル | DeepSeek V4 Flash-0731 |
| メインモデル | K216 REAP / EXL3 3.0 bpw / TP1 |
| ドラフトモデル | K64 DSpark / K5 / 確率的サンプリング |
| コンテキスト長 | 256,000トークン |
| 対応ハードウェア | NVIDIA DGX Spark / GB10 1台 |
| API | OpenAI互換 |

モデル本体、ドラフトモデル、位置補正アダプター、オフライン導入用wheelはHugging Faceで配布しています。このリポジトリでは、DGX Sparkでの起動に必要なランタイムのソースを公開しています。

## 必要環境

- NVIDIA DGX Spark / GB10
- Docker Engine
- Docker Compose v2
- Hugging Face CLI
- 102 GiB以上の空き容量

## セットアップ

### 1. モデルをダウンロードする

```bash
hf download \
  Laplace1313/DeepSeek-V4-Flash-0731-JA-REAP-K216-EXL3-3bpw-DGX-Spark \
  --local-dir ./DeepSeek-V4-Flash-JA-REAP
```

### 2. サーバーを起動する

```bash
cd DeepSeek-V4-Flash-JA-REAP
docker compose -f runtime/compose.example.yml up -d
```

初回起動時は、モデルの重み、AOTアーティファクト、CUDAグラフの読み込みに数分かかります。

### 3. 動作を確認する

```bash
curl -fsS http://127.0.0.1:8009/health
curl -fsS http://127.0.0.1:8009/v1/models
```

## APIの使用例

```bash
curl http://127.0.0.1:8009/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-ja-uncensored-0731",
    "messages": [
      {"role": "user", "content": "日本語で自己紹介してください。"}
    ],
    "temperature": 0,
    "max_tokens": 256
  }'
```

APIサーバーは標準で`127.0.0.1:8009`で待ち受けます。

## 性能

DGX Spark 1台、256Kコンテキスト、OpenAI互換SSEで計測しています。

| 処理 | 平均生成速度 |
|---|---:|
| 日本語の自由文 | 30.02 tok/s |
| コーディング | 33.51 tok/s |
| 構造化JSON | 58.31 tok/s |

構造化JSONは自由文やコードよりドラフトモデルが予測しやすいため、別の処理として掲載しています。

| 評価 | 結果 |
|---|---:|
| MMLU full 5-shot（全57分野・14,042問） | **86.03%**（12,081/14,042） |
| 日本語プロンプト | 50/50 |
| ツール呼び出し | 6/6 |
| HumanEval（50問） | 96% |
| MBPP（50問） | 84% |

MMLUの全条件・カテゴリ別・57分野別結果は[`benchmarks/mmlu-full-5shot-20260827.md`](benchmarks/mmlu-full-5shot-20260827.md)に掲載しています。その他の測定条件は[Hugging Faceのモデルカード](https://huggingface.co/Laplace1313/DeepSeek-V4-Flash-0731-JA-REAP-K216-EXL3-3bpw-DGX-Spark)を参照してください。

## 互換性

次の構成で検証しています。

- NVIDIA DGX Spark / GB10 1台
- 256,000トークンのコンテキスト
- OpenAI互換Chat Completions API
- K64 DSparkによる投機的デコード

起動には付属の専用ランタイムが必要です。標準のTransformers、vLLM、SGLang、llama.cpp、GGUFには対応していません。

## リポジトリ構成

```text
.
├── README.md
├── LICENSE
├── NOTICE.md
├── LICENSES/
└── runtime/
    ├── compose.example.yml
    ├── entrypoint-256k.sh
    ├── entrypoint-toolfix.sh
    ├── serve-ds4-flash.sh
    └── vLLM / SparkInferランタイムソース
```

## ライセンスとクレジット

各ファイルに適用されるライセンス、上流リポジトリ、リビジョン、著作権表示は[`NOTICE.md`](NOTICE.md)にまとめています。ライセンス本文は[`LICENSES/`](LICENSES/)にあります。
