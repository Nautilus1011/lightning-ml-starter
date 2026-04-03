# PyTorch Lightning ガイド

[PyTorch Lightning](https://lightning.ai/) は PyTorch の「定型的なお作法」をまとめたフレームワークです。  
学習ループ・GPU 切り替え・チェックポイント保存などの boilerplate を排除し、モデルのロジックに集中できます。

---

## 2 つの主要クラス

| クラス | 役割 | このリポジトリでの実装 |
|---|---|---|
| `LightningModule` | モデル・損失・オプティマイザの定義 | `src/detection_toolkit/models/detector.py` |
| `LightningDataModule` | データロードの定義 | `src/detection_toolkit/datamodules/voc_datamodule.py` |

---

## LightningModule — `VOCDetector`

```python
# src/detection_toolkit/models/detector.py（抜粋）
import lightning as L

class VOCDetector(L.LightningModule):
    def __init__(self, num_classes=21, lr=0.005, momentum=0.9, weight_decay=0.0005):
        super().__init__()
        self.save_hyperparameters()   # ← チェックポイントにハイパーパラメータを自動保存

        # torchvision の事前学習済みモデルをロード
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        )
        # 出力層をクラス数に合わせて差し替え
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)   # Faster R-CNN は訓練時に loss を返す
        losses = sum(loss_dict.values())
        self.log("train_loss", losses, on_step=True, on_epoch=True, prog_bar=True)
        return losses

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        self.model.train()                        # Faster R-CNN の仕様: eval だと loss が出ない
        with torch.no_grad():
            loss_dict = self.model(images, targets)
        losses = sum(loss_dict.values())
        self.log("val_loss", losses, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.hparams.lr,
            momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return [optimizer], [scheduler]
```

**重要ポイント**

- `self.save_hyperparameters()` を呼ぶと、`__init__` の引数がすべてチェックポイントに保存される。`load_from_checkpoint()` でそのまま復元可能
- `self.log()` を呼ぶだけで wandb / TensorBoard などに自動転送される
- GPU 移動 (`.to(device)`) は Lightning が自動でやるため不要

---

## LightningDataModule — `VOCDataModule`

```python
# src/detection_toolkit/datamodules/voc_datamodule.py（抜粋）
class VOCDataModule(L.LightningDataModule):
    def prepare_data(self):
        # データのダウンロード（1 プロセスだけで実行される）
        datasets.VOCDetection(self.data_dir, year="2012", image_set="train", download=True)

    def setup(self, stage=None):
        # train / val のデータセットを生成（全 GPU で実行される）
        self.train_ds = datasets.VOCDetection(self.data_dir, year="2012", image_set="train",
                                               transform=self.get_transforms())
        self.val_ds   = datasets.VOCDetection(self.data_dir, year="2012", image_set="val",
                                               transform=self.get_transforms())

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, collate_fn=self.collate_fn, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, collate_fn=self.collate_fn, num_workers=self.num_workers)
```

---

## Trainer と Callback

`train.py` では `Trainer` にコールバックを渡すことで、追加コードなしに便利な機能が使えます。

```python
# src/train.py（抜粋）
trainer = L.Trainer(
    max_epochs=cfg.trainer.max_epochs,
    accelerator="auto",          # GPU があれば自動で使う
    precision="16-mixed",        # AMP（自動混合精度）で高速化
    logger=wandb_logger,
    callbacks=[
        ModelCheckpoint(         # val_loss が最良のエポックを自動保存
            monitor="val_loss", mode="min", save_top_k=1,
        ),
        EarlyStopping(           # 改善がなければ自動停止
            monitor="val_loss", patience=3, mode="min",
        ),
        LearningRateMonitor(),   # 学習率の変化を wandb に自動記録
    ],
)
trainer.fit(model, datamodule=dm)
```

| Callback | 何をするか |
|---|---|
| `ModelCheckpoint` | `val_loss` が最小のエポックの重みだけを保存（ディスク節約） |
| `EarlyStopping` | `patience=3` エポック改善がなければ学習を自動停止（過学習防止） |
| `LearningRateMonitor` | 各エポックの学習率を wandb に自動ログ |

---

## チェックポイントからの復元

`save_hyperparameters()` を使っているため、チェックポイントだけあれば再現できます。

```python
from detection_toolkit.models.detector import VOCDetector

# ハイパーパラメータも一緒に復元される
model = VOCDetector.load_from_checkpoint("outputs/train/.../best-checkpoint.ckpt")
model.eval()
```
