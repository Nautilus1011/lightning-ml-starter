---
name: nbexec
description: "Docker の dev コンテナ内で Jupyter notebook を実行し、結果を outputs/notebooks/ に保存する"
argument-hint: "<notebook_path> [--output <output_filename>] [--timeout <seconds>]"
---

# /nbexec

Docker の `dev` コンテナ内で指定した notebook を実行し、結果を `outputs/notebooks/` に保存する。
元の notebook は書き換えない。

## 使い方

```text
/nbexec notebooks/training_and_inference.ipynb
/nbexec notebooks/training_and_inference.ipynb --output check_executed.ipynb
/nbexec notebooks/training_and_inference.ipynb --timeout 3600
```

## 手順

1. notebook のパスと出力ファイル名を決める（省略時は `<元の名前>_executed.ipynb`）
2. `docker compose ps` で `dev` コンテナが起動していることを確認する
3. 以下を実行する

```bash
docker compose exec dev bash -lc "jupyter nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.timeout=<timeout> \
  --output-dir outputs/notebooks \
  --output <output_name> \
  <notebook_path>"
```

4. 出力先 `outputs/notebooks/<output_name>` を報告する
5. エラー時は、どのセルで止まったか、import エラーかカーネルエラーかを切り分けて報告する

## 注意

- `notebooks/training_and_inference.ipynb` は学習を含むため長時間かかる。`--timeout` を十分に取る
- `outputs/notebooks/` はローカル管理のみ（Git 管理外）
- 実行結果を `notebooks/` の証跡として差し替える場合は、ユーザーに確認してから行う
