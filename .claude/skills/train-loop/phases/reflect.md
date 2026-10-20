# Phase 5: REFLECT

← [SKILL.md に戻る](../SKILL.md)

結果を評価し、次のループに必要な記録を残す。

## Step 1: メトリクスの確認

ベストの `val_loss` は、`ModelCheckpoint` がチェックポイントに保存した値から読む（W&B がオフラインでも読める）。

```bash
docker compose exec dev bash -lc "python -c \"
import torch
ckpt = torch.load('<current_run_dir>/checkpoints/best-checkpoint.ckpt', map_location='cpu', weights_only=False)
print('epoch:', ckpt['epoch'])
for key, state in ckpt['callbacks'].items():
    if key.startswith('ModelCheckpoint'):
        print('best_model_score:', float(state['best_model_score']))
\""
```

あわせて以下も確認する。

- 実際に使われた設定: `<current_run_dir>/.hydra/config.yaml` と `overrides.yaml`
- 最終エポックの値: `<current_run_dir>/wandb/latest-run/files/wandb-summary.json`
- EarlyStopping で止まったかどうか: 学習ログ

## Step 2: ベスト更新の判定

`BRIEF.md` の `metric_goal`（`val_loss` なら `minimize`）に従い、`state.json` の `best_metric` と比べる。

## Step 3: LOOP_LOG.md の更新

- `## Key Results`: ベストを更新した時だけ追記
- `## Recent Decisions`: 毎ループ追記

```text
[01-01 14:30] Loop 2: model.lr=0.01, val_loss=0.412 (epoch 7) ← NEW BEST
[01-01 14:30] Loop 2: lr を上げて改善。次は momentum=0.95 を試す。
```

## Step 4: KNOWLEDGE.md の更新

複数のループで確認できた傾向だけを追記する。1 回だけの結果は「仮説」として書き分ける。

## Step 5: state.json の更新

```json
{
  "status": "idle",
  "loop_number": 2,
  "best_metric": 0.412,
  "best_loop": 2,
  "consecutive_crashes": 0,
  "current_run_dir": null
}
```

停止条件を確認し、満たしていなければ THINK に戻る。
