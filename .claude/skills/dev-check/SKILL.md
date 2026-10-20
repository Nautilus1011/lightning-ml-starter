---
name: dev-check
description: "PR 前の動作確認を、変更内容に応じた範囲で行い結果を報告する"
---

# /dev-check

PR 前の動作確認を行う。

## 1. 変更ファイルの確認

```bash
git diff main...HEAD --name-only
git status --short
```

## 2. 変更内容に応じたチェック

| 変更した場所 | チェック |
|---|---|
| `src/` の Python | `python3 -m compileall src` は必須。可能なら学習・推論の疎通確認 |
| `configs/` | 学習の疎通確認で、override が意図どおり効くか確認 |
| `Dockerfile` / `docker-compose.yml` / `requirements-dev.txt` | `docker compose build` が通るか確認 |
| `README.md` / `docs/` / `CLAUDE.md` / `.claude/` | 書かれたコマンドやパスが実在するか、記述同士が矛盾しないか確認 |

### 学習の疎通確認

1 エポックだけ回して、学習→検証→チェックポイント保存まで通るかを見る。
VOC の train 全体を 1 周するため、GPU がない環境では時間がかかる。

```bash
docker compose exec dev bash -lc "WANDB_MODE=offline PYTHONPATH=src python src/train.py \
  trainer.max_epochs=1 \
  logger.experiment_name=dev-check \
  hydra.run.dir=outputs/dev-check/\$(date +%Y%m%d-%H%M%S)"
```

### 推論の疎通確認

```bash
docker compose exec dev bash -lc "PYTHONPATH=src python src/inference.py \
  --checkpoint <ckpt のパス> \
  --image data/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg"
```

## 3. 結果報告

通過したもの・省略したものを、理由とあわせて明示する。

```text
OK:   python3 -m compileall src
OK:   docs のコマンドとパスの整合性
SKIP: 学習の疎通確認（GPU がなく、1 エポックに時間がかかるため）
```

- 省略した項目は PR の `## 補足情報` にもそのまま書く
- 失敗した項目を「おそらく問題ない」とまとめない。エラー内容をそのまま報告する
