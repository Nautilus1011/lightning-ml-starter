# Phase 4: MONITOR

← [SKILL.md に戻る](../SKILL.md)

学習中の状態を軽く確認する。頻繁に確認しすぎない（目安: 10〜20 分おき）。

## 手順

1. `state.json` を `status: monitoring` に更新する
2. ログの末尾で進み具合とエラーの有無を確認する

```bash
tail -n 20 workspace/train-loop/logs/loop-<N>.log
```

3. 出力先の生成物を確認する

```bash
ls -la <current_run_dir> <current_run_dir>/checkpoints
```

4. GPU 環境なら `nvidia-smi` でメモリ使用量を確認する

## 判定

| 状態 | 判断 | 次の行動 |
| :-- | :-- | :-- |
| プロセスが終了し、`checkpoints/best-checkpoint.ckpt` がある | 正常終了 | `status: reflecting` にして REFLECT へ |
| `Traceback` や `CUDA out of memory` がある | エラー終了 | `consecutive_crashes` を 1 増やし、原因を `LOOP_LOG.md` に書いて THINK へ |
| `val_loss` が `nan` | guardrail 違反 | 学習を止め、ユーザーに報告する |
| 実行中 | 継続 | 次の確認時刻を決めて待つ |
