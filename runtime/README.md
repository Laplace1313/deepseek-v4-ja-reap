# Runtime source mirror

このdirectoryには、Hugging Face model packageをDGX Sparkで起動するためにComposeがbind-mountするruntime sourceを置いています。

検証済み構成：

```text
image: ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4
vLLM: 0.15.1+nv26.2
xgrammar: 0.2.4
transformers: 5.13.1
```

このGit mirrorだけでは起動できません。Model weight、K64 draft、position adapter、checksum付きoffline wheelsは[Hugging Face model package](https://huggingface.co/Laplace1313/DeepSeek-V4-Flash-0731-JA-REAP-K216-EXL3-3bpw-DGX-Spark)から取得してください。HF package内の`runtime/compose.example.yml`を使うのが正式な起動方法です。

これらのsourceを別versionのvLLMやSparkInferへそのまま適用しないでください。Composeのmount先を含めて互換性contractです。
