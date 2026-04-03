# Weights & Biases (wandb) ガイド

[Weights & Biases](https://wandb.ai/) は学習曲線・予測画像・ハイパーパラメータをクラウドで管理する実験管理ツールです。  
このリポジトリでは学習・推論の両方で wandb に自動ログが送られます。

---

## セットアップ

### 1. アカウント作成

1. [https://wandb.ai/site](https://wandb.ai/site) → **Sign up**
2. ログイン後: 右上アバター → **User Settings** → **API keys** → **New key** でキーをコピー

### 2. コンテナ内でログイン

```bash
wandb login
# プロンプトに API キーを貼り付けて Enter
# "Successfully logged in to Weights & Biases!" と表示されれば完了
```

---

## 学習時のログ

`train.py` では `WandbLogger` を使って Lightning の `self.log()` を自動転送します。

```python
# src/train.py（抜粋）
from lightning.pytorch.loggers import WandbLogger

wandb_logger = WandbLogger(
    project="voc-object-detection",
    name="fasterrcnn_mobilenet_v3_320",
)

trainer = L.Trainer(logger=wandb_logger, ...)
```

モデル側では `self.log()` を呼ぶだけで wandb に届きます。

```python
# src/detection_toolkit/models/detector.py（抜粋）
def training_step(self, batch, batch_idx):
    ...
    self.log("train_loss", losses, on_step=True, on_epoch=True, prog_bar=True)
```

### ダッシュボードに記録される内容

| 項目 | 説明 |
|---|---|
| `train_loss` / `val_loss` | 学習・検証ロスのグラフ |
| `train_loss_classifier` など | Faster R-CNN の損失内訳（分類・回帰・RPN） |
| `lr-SGD` | 学習率の変化（`LearningRateMonitor` が自動記録） |
| 予測画像 | バウンディングボックス付き（検証バッチの先頭 1 枚、スコア > 0.5） |

---

## 推論時のログ

`inference.py` でも wandb run が作られ、推論結果が記録されます。

```python
# src/inference.py（抜粋）
wandb.init(project="voc-object-detection", job_type="inference")

wandb_img = wandb.Image(
    save_path,
    caption=f"{filename} — {n_detected} objects",
    boxes={"predictions": {"box_data": wandb_boxes, "class_labels": class_labels}},
)
wandb.log({"predictions": wandb_img})
```

---

## オフラインモードでのデバッグ

wandb への接続なしに動作確認したい場合は `WANDB_MODE=offline` を指定します。

```bash
WANDB_MODE=offline PYTHONPATH=src python src/train.py
```

ローカルに `.wandb` ファイルが保存され、後から `wandb sync` でアップロードできます。

---

## 参考: wandb Sweep（ハイパーパラメータ自動探索）

複数の設定を自動で試して最良のパラメータを探す **Sweep** 機能については、  
別ドキュメント [wandb_sweep.md](wandb_sweep.md) を参照してください。
