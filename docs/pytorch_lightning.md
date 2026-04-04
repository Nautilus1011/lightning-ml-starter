# PyTorch Lightning ガイド

[PyTorch Lightning](https://lightning.ai/) は PyTorch の「定型的なお作法」を肩代わりするフレームワークです。  
学習ループ・GPU 切り替え・チェックポイント保存などの boilerplate を排除し、モデルのロジックに集中できます。

---

## PyTorch との比較

素の PyTorch では学習ループをすべて自分で書く必要があります。

```python
# 素の PyTorch（自分で書く必要がある定型処理）
for epoch in range(max_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
    # GPU 指定、AMP、チェックポイント保存... すべて自前実装
```

Lightning はこれらを内部で処理し、**「何を学ぶか」だけをコードに残す**設計になっています。

---

## 2 つの主要クラス

| クラス | 役割 | このリポジトリでの実装 |
|---|---|---|
| `LightningModule` | モデル・損失関数・オプティマイザの定義 | `src/detection_toolkit/models/detector.py` |
| `LightningDataModule` | データセット・DataLoader の定義 | `src/detection_toolkit/datamodules/voc_datamodule.py` |

---

## LightningModule — `VOCDetector`

モデルに関する「すべてのロジック」を 1 クラスにまとめます。

### `__init__` — モデルの構築と `save_hyperparameters()`

```python
class VOCDetector(L.LightningModule):
    def __init__(self, num_classes=21, lr=0.005, momentum=0.9, weight_decay=0.0005):
        super().__init__()
        self.save_hyperparameters()  # __init__ の引数をチェックポイントに自動保存

        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        )
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

`self.save_hyperparameters()` が **最も重要なメソッドの一つ**です。

- `__init__` に渡した `num_classes`, `lr` などが `.ckpt` ファイルに自動保存される
- `load_from_checkpoint()` 呼び出し時に引数を指定しなくても完全に復元できる
- `self.hparams.lr` のようにアクセスできる

```python
# 再現例：チェックポイントだけあればモデルを完全復元できる
model = VOCDetector.load_from_checkpoint("best-checkpoint.ckpt")
```

---

### `training_step` — 1 ステップの学習処理

```python
def training_step(self, batch, batch_idx):
    images, targets = batch
    loss_dict = self.model(images, targets)  # Faster R-CNN は学習時に loss 辞書を返す
    losses = sum(loss_dict.values())
    self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True)
    self.log("train_loss", losses, on_step=True, on_epoch=True, prog_bar=True)
    return losses
```

`self.log()` / `self.log_dict()` を呼ぶだけで wandb などのロガーに自動転送されます。  
`optimizer.zero_grad()` や `loss.backward()`, `optimizer.step()` は Lightning が自動で行います。

| `log()` 引数 | 意味 |
|---|---|
| `on_step=True` | ステップごとにロギング（グラフが細かい） |
| `on_epoch=True` | エポック末にエポック平均をロギング |
| `prog_bar=True` | プログレスバーに表示 |

---

### `validation_step` — 1 ステップの検証処理

```python
def validation_step(self, batch, batch_idx):
    images, targets = batch
    self.model.train()          # Faster R-CNN の仕様: eval モードでは loss が計算されない
    with torch.no_grad():
        loss_dict = self.model(images, targets)
    losses = sum(loss_dict.values())
    self.log("val_loss", losses, on_epoch=True, prog_bar=True)
```

通常は `model.eval()` で検証しますが、Faster R-CNN は `eval` モードだと損失を返さない仕様です。  
そのため `model.train()` のまま `torch.no_grad()` で囲んで損失を計算しています。

> `torch.no_grad()` はバックプロパゲーションの計算グラフを作らない宣言です。  
> これにより推論時のメモリ使用量を大幅に削減できます。

---

### `configure_optimizers` — オプティマイザの設定

```python
def configure_optimizers(self):
    optimizer = torch.optim.SGD(
        self.parameters(),
        lr=self.hparams.lr,
        momentum=self.hparams.momentum,
        weight_decay=self.hparams.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    return [optimizer], [lr_scheduler]
```

- `self.parameters()` : モデルの全パラメータを返す PyTorch 標準メソッド
- `self.hparams.lr` : `save_hyperparameters()` で保存した値へのアクセス
- `StepLR` : `step_size` エポックごとに学習率を `gamma` 倍する。学習が停滞するのを防ぐ
- リストで `[optimizer], [scheduler]` を返すのが Lightning の規約（複数設定への拡張に対応するため）

---

## LightningDataModule — `VOCDataModule`

データに関する処理を 1 クラスにまとめます。  
`LightningModule` から分離することで、モデルとデータの結合度を下げられます。

### ライフサイクルメソッド

| メソッド | 呼ばれるタイミング | 役割 |
|---|---|---|
| `prepare_data()` | 1 プロセスのみ・1 回だけ | データのダウンロードなど（状態を持たせない） |
| `setup(stage)` | 全プロセス・Trainer 開始前 | データセットオブジェクトの生成 |
| `train_dataloader()` | 学習開始時 | 学習用 DataLoader を返す |
| `val_dataloader()` | 検証開始時 | 検証用 DataLoader を返す |

```python
def prepare_data(self):
    # 分散学習で複数プロセスが同時にダウンロードしないよう、1 プロセスだけで実行される
    datasets.VOCDetection(self.data_dir, year="2012", image_set="train", download=True)
    datasets.VOCDetection(self.data_dir, year="2012", image_set="val", download=True)

def setup(self, stage=None):
    # prepare_data() の後に全プロセスで呼ばれる。データセットオブジェクトを生成する
    self.train_ds = datasets.VOCDetection(self.data_dir, year="2012", image_set="train",
                                           transform=self.get_transforms())
    self.val_ds   = datasets.VOCDetection(self.data_dir, year="2012", image_set="val",
                                           transform=self.get_transforms())
```

> `prepare_data()` と `setup()` を分けている理由:  
> マルチGPU学習では `setup()` は各プロセスで実行されますが、`prepare_data()` は 1 回だけ実行されます。  
> ダウンロード処理を `setup()` に書くと複数プロセスが同時にファイルを書き込もうとして壊れます。

---

### `collate_fn` — バッチの作り方

```python
def collate_fn(self, batch):
    images, targets = [], []
    for img, label in batch:
        images.append(img)
        objs = label['annotation']['object']
        if not isinstance(objs, list):
            objs = [objs]
        boxes  = [[float(o['bndbox'][k]) for k in ('xmin','ymin','xmax','ymax')] for o in objs]
        labels = [self.class_to_idx[o['name']] for o in objs]
        targets.append({
            "boxes":  torch.as_tensor(boxes,  dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
        })
    return images, targets
```

通常の DataLoader は画像を `(N, C, H, W)` の Tensor にスタックしますが、物体検出では  
**画像ごとにボックス数が異なる**ため、単純なスタックができません。  
`collate_fn` を渡すことで「Tensor ではなくリストでまとめる」挙動に変更しています。

---

## Trainer — 学習の実行エンジン

`Trainer` は学習・検証ループを管理する Lightning の中核クラスです。

```python
trainer = L.Trainer(
    max_epochs=cfg.trainer.max_epochs,
    accelerator="auto",       # GPU / CPU / TPU を自動選択
    devices="auto",           # 利用可能なデバイス数を自動検出
    precision="16-mixed",     # 混合精度学習（FP16+FP32）でメモリ削減・高速化
    logger=wandb_logger,      # WandB にメトリクスを転送
    callbacks=[...],
    log_every_n_steps=10,
)
trainer.fit(model, datamodule=dm)
```

`trainer.fit()` を呼ぶだけで以下がすべて自動で行われます。

- 学習ループ（forward → loss → backward → optimizer step）
- 検証ループ（`val_dataloader` のエポック末実行）
- GPU へのデータ・モデル転送
- 混合精度（AMP）の制御
- コールバックの呼び出し

---

## コールバック — 学習ループへのフック

コールバックは学習ループの特定タイミングに自動で呼ばれる処理です。  
`Trainer` に渡すだけで機能します。

```python
callbacks=[
    ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        monitor="val_loss", mode="min", save_top_k=1,
        filename="best-checkpoint",
    ),
    EarlyStopping(
        monitor="val_loss", patience=3, mode="min",
    ),
    LearningRateMonitor(logging_interval='epoch'),
]
```

| コールバック | 動作 |
|---|---|
| `ModelCheckpoint` | `val_loss` が最小のエポックの重みだけを保存する。`save_top_k=1` でディスクを節約 |
| `EarlyStopping` | `patience=3` エポック以上改善がなければ学習を自動停止する。過学習防止にも有効 |
| `LearningRateMonitor` | 各エポックの学習率を自動でロガーに記録する |

---

## チェックポイントからの復元

```python
from detection_toolkit.models.detector import VOCDetector

model = VOCDetector.load_from_checkpoint("outputs/train/.../best-checkpoint.ckpt")
model.eval()
```

`save_hyperparameters()` を使っているため、`num_classes` などの引数を指定しなくても  
チェックポイントファイルだけで完全にモデルを復元できます。
