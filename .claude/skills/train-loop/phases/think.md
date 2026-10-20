# Phase 0 & 1: 初期化・THINK

← [SKILL.md に戻る](../SKILL.md)

## Phase 0: 初期化

### 通常起動

1. `workspace/train-loop/BRIEF.md` を読み、必須項目がそろっているか確認する
2. `workspace/train-loop/state.json` を読む。なければ作る
3. 引数と `BRIEF.md` から停止条件を確定する（引数が優先）
4. `started_at` を記録し、`loop_number: 0` から開始する

```json
{
  "status": "idle",
  "loop_number": 0,
  "started_at": "2026-01-01T12:00:00",
  "best_metric": null,
  "best_loop": null,
  "consecutive_crashes": 0,
  "current_run_dir": null
}
```

### `--resume`

既存の `state.json` の `loop_number` から再開する。`status` が `training` のまま止まっていたら、まず MONITOR で前回の学習の結末を確認する。

### `--new-brief`

1. `workspace/train-loop/archive/<日付>-<experiment_name>/` を作る
2. 既存の `LOOP_LOG.md`・`state.json`・`BRIEF.md` を退避する
3. `KNOWLEDGE.md` に、確定した知見と試していない案を追記する（`KNOWLEDGE.md` は退避しない）
4. 新しい `BRIEF.md` をユーザーと作り、`LOOP_LOG.md` と `state.json` を初期化する

## Phase 1: THINK

1. `HUMAN_DIRECTIVE.md` があれば最優先で読む。反映したら `LOOP_LOG.md` に記録し、ファイルを削除する
2. `BRIEF.md`・`KNOWLEDGE.md`・`LOOP_LOG.md` を読む
3. 以下を決めて `LOOP_LOG.md` に書く
   - 仮説（何を変えると、なぜ良くなると考えるか）
   - Hydra override（`allowed_overrides` の範囲内で、前回からの差分は原則 1 つ）
   - run 名と出力先
4. `state.json` を `status: planning` に更新する

### よく触る override（`configs/train.yaml`）

| キー | 既定値 | 備考 |
| :-- | :-- | :-- |
| `model.lr` | `0.005` | SGD の学習率 |
| `model.momentum` | `0.9` | |
| `model.weight_decay` | `0.0005` | |
| `data.batch_size` | `8` | GPU メモリに注意 |
| `data.num_workers` | `4` | |
| `trainer.max_epochs` | `10` | |
| `callbacks.early_stopping.patience` | `3` | |

### 事前確認

- override のキーが `configs/train.yaml` に実在するか（存在しないキーは Hydra がエラーにする）
- 出力先が `outputs/train-loop/` 配下で、過去のループと重複しないか
