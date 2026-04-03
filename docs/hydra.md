# Hydra による設定管理

[Hydra](https://hydra.cc/) は設定ファイルの管理とコマンドラインオーバーライドを簡単にするライブラリです。  
このリポジトリでは `configs/train.yaml` が学習のすべてのパラメータを一元管理しています。

---

## 設定ファイルの場所

```
configs/
└── train.yaml   # 学習パラメータの定義
```

## `configs/train.yaml` の全体像

```yaml
hydra:
  run:
    dir: outputs/train/${now:%Y-%m-%d}/${now:%H-%M-%S}   # 実行ごとに自動でディレクトリを作成

logger:
  project_name: "voc-object-detection"
  experiment_name: "fasterrcnn_mobilenet_v3_320"

data:
  data_dir: "${hydra:runtime.cwd}/data"
  batch_size: 8
  num_workers: 4

model:
  num_classes: 21    # VOC 20クラス + 背景
  lr: 0.005
  momentum: 0.9
  weight_decay: 0.0005

trainer:
  max_epochs: 10
  accelerator: "auto"
  devices: "auto"
  precision: "16-mixed"
  log_every_n_steps: 10

callbacks:
  early_stopping:
    monitor: "val_loss"
    patience: 3
    mode: "min"
  model_checkpoint:
    monitor: "val_loss"
    save_top_k: 1
    mode: "min"
    filename: "best-checkpoint"
```

---

## `train.py` での使い方

`@hydra.main` デコレータを付けるだけで、yaml の内容が `cfg` として受け取れます。

```python
# src/train.py（抜粋）
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def train(cfg: DictConfig):
    # cfg.data.batch_size のように .区切りでアクセス
    dm = VOCDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    model = VOCDetector(
        num_classes=cfg.model.num_classes,
        lr=cfg.model.lr,
    )
```

---

## コマンドラインからパラメータを変更する

yaml ファイルを書き換えなくても、実行時に `キー=値` で上書きできます。

```bash
# 学習率を変更
PYTHONPATH=src python src/train.py model.lr=0.001

# バッチサイズとエポック数を同時に変更
PYTHONPATH=src python src/train.py data.batch_size=4 trainer.max_epochs=5

# num_workers=0（Notebook / Docker 環境向け）
PYTHONPATH=src python src/train.py data.num_workers=0
```

---

## 出力ディレクトリの自動生成

`hydra.run.dir` に `${now:%Y-%m-%d}/${now:%H-%M-%S}` を指定しているため、  
実行するたびにタイムスタンプ付きのディレクトリが自動で作られます。

```
outputs/train/
└── 2026-04-03/
    └── 02-48-00/
        ├── .hydra/              ← この実行の設定スナップショット（再現性の担保）
        │   ├── config.yaml
        │   └── overrides.yaml   ← コマンドラインで変えた値だけ記録される
        ├── checkpoints/
        │   └── best-checkpoint.ckpt
        └── train.log
```

**`.hydra/overrides.yaml` の例**

```yaml
# model.lr=0.001 を指定した場合
- model.lr=0.001
```

これにより「どの設定で動かしたか」が常に記録され、再現が容易になります。

---

## 設定値をコードから参照する

`HydraConfig.get()` を使うと、実行時に Hydra が決定した値（出力ディレクトリなど）も取得できます。

```python
from hydra.core.hydra_config import HydraConfig

run_dir = Path(HydraConfig.get().runtime.output_dir)
# → outputs/train/2026-04-03/02-48-00 のような絶対パス
```

このリポジトリでは `train.py` がこの方法でチェックポイントの保存先を決めています。
