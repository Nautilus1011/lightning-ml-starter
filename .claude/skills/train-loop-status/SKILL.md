---
name: train-loop-status
description: "実行中または完了した train-loop の状態を確認して表示する"
---

# /train-loop-status

`/train-loop` の状態を表示する。ファイルを読むだけで、何も変更しない。

## 動作

1. `workspace/train-loop/BRIEF.md` を読む
2. `workspace/train-loop/state.json` を読む
3. `workspace/train-loop/LOOP_LOG.md` を読む
4. `workspace/train-loop/HUMAN_DIRECTIVE.md` の有無を確認する
5. `state.json` の `status` が `training` / `monitoring` の場合:
   - `workspace/train-loop/logs/loop-<N>.log` の末尾 20 行を表示する
   - `nvidia-smi` を実行できる環境なら GPU 状態を表示する

## 出力フォーマット

```markdown
# train-loop ステータス

## 実験目標
<BRIEF.md の Goal>

## 進捗
- 完了ループ数: N / max_loops
- ステータス: idle / planning / training / monitoring / reflecting / stopped
- ベスト val_loss: X.XXX (Loop N)
- 経過時間: H.H 時間
- 連続クラッシュ数: 0

## 最新の学習ログ
<ログ末尾 20 行>

## GPU ステータス
<nvidia-smi の出力、または unavailable>

## Key Results
<LOOP_LOG.md の Key Results>

## Recent Decisions
<LOOP_LOG.md の Recent Decisions>

## 未処理の指示
None（または HUMAN_DIRECTIVE.md の内容）
```
