# WandB Sweep — ハイパーパラメータ自動探索

WandB Sweep は複数のハイパーパラメータ設定を自動で試し、最良の組み合わせを探す機能です。

---

## 仕組み

```
sweep.yaml を登録  →  Sweep ID が発行される  →  Agent が自動で run を繰り返す
```

- **Controller**: どの設定を次に試すか決める（wandb サーバー側）
- **Agent**: 設定を受け取って学習を 1 回実行するプロセス

---

## Step 1 — sweep.yaml を作成

```yaml
# src/sweep.yaml
program: src/train.py
method: bayes          # bayes（ベイズ最適化） / random / grid から選択
metric:
  name: val_loss
  goal: minimize
parameters:
  model.lr:
    distribution: log_uniform_values
    min: 1.0e-4
    max: 1.0e-1
  data.batch_size:
    values: [4, 8]
  model.weight_decay:
    values: [0.0001, 0.0005, 0.001]
```

**`method` の選び方**

| method | 特徴 | 向いている場面 |
|---|---|---|
| `bayes` | 前の結果をもとに次を決める（効率的） | パラメータ数が多い・探索コストが高い |
| `random` | ランダムサンプリング | 初期探索・並列実行 |
| `grid` | 全組み合わせを試す | パラメータ候補が少ない |

---

## Step 2 — Sweep を登録

```bash
cd /app
PYTHONPATH=src wandb sweep src/sweep.yaml
# → wandb: Creating sweep with ID: abc123
# → wandb: View sweep at: https://wandb.ai/<entity>/voc-object-detection/sweeps/abc123
```

---

## Step 3 — Agent を起動

```bash
# 発行された sweep ID を使う
PYTHONPATH=src wandb agent <entity>/voc-object-detection/<sweep-id>
```

Agent は wandb サーバーから設定を受け取り、終わると次の設定を要求し続けます。  
`--count` で実行回数を制限できます。

```bash
# 最大 10 回で止める
PYTHONPATH=src wandb agent <entity>/voc-object-detection/<sweep-id> --count 10
```

---

## Hydra との組み合わせ注意点

このリポジトリの `train.py` は `@hydra.main` を使っているため、Sweep がコマンドライン引数で渡す  
`model.lr=0.001` のような値がそのまま Hydra のオーバーライドとして機能します。  
追加の実装は不要です。

---

## 結果の確認

Sweep 終了後、wandb ダッシュボードの **Sweeps** タブで以下が確認できます。

- 各 run のメトリクス比較テーブル
- パラメータの重要度（Importance）
- 平行座標プロット（どの組み合わせが良かったか）
