---
name: train-loop
description: "src/train.py に Hydra override を切り替えながら、計画→学習→振り返りのループを回して記録する"
argument-hint: "[--loops N] [--hours H] [--target-metric X] [--resume] [--skip-dry-run] [--new-brief]"
---

# /train-loop

`src/train.py` に対して Hydra override を切り替えながら、学習ループを回す。
1 ループは THINK（計画）→ DRY-RUN → EXECUTE（学習）→ MONITOR → REFLECT（振り返り）で構成する。
人が決めた目標と制約（`BRIEF.md`）の範囲内でだけ試行し、判断の経緯をすべてファイルに残す。

## 使い方

```text
/train-loop                      # BRIEF.md の設定で開始
/train-loop --loops 5            # 最大 5 ループ
/train-loop --hours 6            # 最大 6 時間
/train-loop --target-metric 0.40 # val_loss が 0.40 以下になったら停止
/train-loop --resume             # state.json から再開
/train-loop --skip-dry-run       # DRY-RUN を省略
/train-loop --new-brief          # 過去ログを退避して新しい実験を始める
```

## 作業ファイル

すべて `workspace/train-loop/` に置く（Git 管理外）。

| ファイル | 役割 | 書く人 |
| :-- | :-- | :-- |
| `BRIEF.md` | 実験目標・制約 | 人（なければ聞き取ってエージェントが下書きし、人が承認する） |
| `HUMAN_DIRECTIVE.md` | ループ中の人からの指示 | 人 |
| `LOOP_LOG.md` | ループごとの結果と判断 | エージェント |
| `KNOWLEDGE.md` | 実験をまたいで使える知見 | エージェント |
| `state.json` | ループの状態 | エージェント |
| `logs/loop-<N>.log` | 学習の標準出力 | エージェント |
| `archive/` | 過去の実験ログ | エージェント |

## BRIEF.md に必須の項目

```markdown
## Goal
<何を改善したいか>

- experiment_name: lr-search
- metric_name: val_loss
- metric_goal: minimize
- max_loops: 5
- max_hours: 6

## allowed_overrides
<試してよい設定。例: model.lr, model.momentum, data.batch_size, trainer.max_epochs>

## frozen_constraints
<変えてはいけない条件。例: モデル構造、データセット、trainer.precision>

## guardrails
<中断すべき状態。例: val_loss が NaN、GPU メモリ不足が続く>
```

`BRIEF.md` がない、または必須項目が欠けている場合は、ループを始めずにユーザーと BRIEF を作る。

## 実行フロー

> 各フェーズに入る前に、対応する `phases/` のファイルを読むこと。

| フェーズ | ファイル |
| :-- | :-- |
| 初期化・THINK | `phases/think.md` |
| MONITOR | `phases/monitor.md` |
| REFLECT | `phases/reflect.md` |

### DRY-RUN

本番の前に、同じ override で 1 エポックだけ回して end-to-end で通るか確かめる。

```bash
docker compose exec dev bash -lc "WANDB_MODE=offline PYTHONPATH=src python src/train.py \
  trainer.max_epochs=1 \
  logger.experiment_name=dryrun-loop-<N> \
  hydra.run.dir=outputs/train-loop/dryrun-loop-<N> \
  <THINK で決めた override>"
```

### EXECUTE

```bash
docker compose exec dev bash -lc "PYTHONPATH=src python src/train.py \
  logger.experiment_name=<experiment_name>-loop-<N> \
  hydra.run.dir=outputs/train-loop/<experiment_name>/loop-<N> \
  <THINK で決めた override>" 2>&1 | tee workspace/train-loop/logs/loop-<N>.log
```

- 学習はバックグラウンドで実行し、終了を待つ間は MONITOR を行う
- 実行前に `state.json` を更新する

```json
{
  "status": "training",
  "loop_number": 1,
  "current_run_dir": "outputs/train-loop/<experiment_name>/loop-1"
}
```

### 停止条件

いずれかを満たしたら停止し、ユーザーに結果を報告する。

- `loop_number >= max_loops`
- 経過時間 `>= max_hours`
- `best_metric` が目標に到達
- `consecutive_crashes >= 3`
- `guardrails` に該当する状態になった

## 人間の介入

ループ中に方針を変えたい時は、人が指示ファイルを置く。次の THINK で最優先に読む。

```bash
echo "次は lr を 0.001 にして試してください" > workspace/train-loop/HUMAN_DIRECTIVE.md
```

## ルール

- `allowed_overrides` 以外の設定を変えない。コードの変更もしない
- `frozen_constraints` に触れる変更が必要だと判断したら、ループを止めてユーザーに相談する
- 結果は実際に読み取った値だけを記録する。推測で埋めない
- 終了後、結果を残す価値があれば `.claude/rules/experiment-report.md` に沿って `EXPERIMENT.md` にまとめるか、ユーザーに提案する
