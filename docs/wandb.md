# Weights & Biases (wandb) ガイド

[Weights & Biases](https://wandb.ai/)（wandb）は機械学習の実験管理ツールです。  
学習曲線・ハイパーパラメータ・予測画像などをクラウドのダッシュボードで記録・比較できます。

「どの設定で学習したか」「あの実験の val_loss はいくつだったか」といった情報を  
**手動で記録しなくても自動で管理できる**のが最大の利点です。

---

## wandb の基本概念

| 用語 | 意味 |
|---|---|
| **Project** | 実験のグループ。1 つのタスク（例: VOC 物体検出）に対して 1 つ作る |
| **Run** | 1 回の実行（学習・推論など）。Project 内に蓄積される |
| **Config** | その Run のハイパーパラメータ。`lr`, `batch_size` など |
| **Log** | ステップ・エポックごとに記録する値。`loss`, `accuracy` などのグラフになる |
| **Artifact** | モデルファイル・データセットなどのバージョン管理されたファイル |

---

## セットアップ

### 1. アカウント作成と API キーの取得

1. [https://wandb.ai/site](https://wandb.ai/site) → **Sign up**
2. ログイン後: 右上アバター → **User Settings** → **API keys** → **New key** でキーをコピー

### 2. コンテナへの API キーの設定

プロジェクトルートに `.env` ファイルを作成します（`docker compose` が自動で読み込みます）。

```bash
# ホスト側で実行
echo "WANDB_API_KEY=your_api_key_here" > .env
```

`.env` は `.gitignore` に追加し、git に含めないようにしてください。

### 3. コンテナ内でのログイン（初回）

```bash
wandb login
# プロンプトに API キーを貼り付けて Enter
# "Successfully logged in to Weights & Biases!" と表示されれば完了
```

---

## 学習時のロギング（`train.py`）

### `WandbLogger` の設定

```python
from lightning.pytorch.loggers import WandbLogger

wandb_logger = WandbLogger(
    project=cfg.logger.project_name,      # ダッシュボードのプロジェクト名
    name=cfg.logger.experiment_name,      # この Run の名前
    save_dir=str(run_dir),                # wandb のローカルログの保存先
)
```

`WandbLogger` を `Trainer` の `logger` に渡すだけで、`self.log()` の値がすべて自動転送されます。  
個別に `wandb.log()` を呼ぶ必要はありません。

- `project` : 同じプロジェクト内の Run は比較グラフで並べて見られる
- `name` : 実験の内容を表す識別名（省略するとランダムな名前が付く）
- `save_dir` : Hydra の出力ディレクトリを指定することで、実験ごとのディレクトリにログが収まる

### `self.log()` によるメトリクスの記録

```python
# src/detection_toolkit/models/detector.py
def training_step(self, batch, batch_idx):
    ...
    self.log("train_loss", losses, on_step=True, on_epoch=True, prog_bar=True)

def validation_step(self, batch, batch_idx):
    ...
    self.log("val_loss", losses, on_epoch=True, prog_bar=True)
```

`self.log()` を呼ぶだけで WandbLogger が受け取り、ダッシュボードのグラフに反映されます。

| 引数 | 効果 |
|---|---|
| `on_step=True` | ステップ（バッチ）ごとに記録。グラフが細かくなる |
| `on_epoch=True` | エポック末にエポック平均を記録 |
| `prog_bar=True` | ターミナルのプログレスバーに表示 |

### `LearningRateMonitor` による学習率の自動記録

```python
from lightning.pytorch.callbacks import LearningRateMonitor

callbacks=[LearningRateMonitor(logging_interval='epoch')]
```

コールバックを渡すだけで、各エポックの学習率が自動的に wandb に記録されます。  
`StepLR` などでスケジューラを使う場合、学習率が意図通りに変化しているか確認できます。

### ダッシュボードに記録される内容

| 項目 | 記録タイミング | ソース |
|---|---|---|
| `train_loss` | ステップ・エポックごと | `training_step` の `self.log()` |
| `train_loss_classifier` など | ステップ・エポックごと | `self.log_dict()` による損失内訳 |
| `val_loss` | エポックごと | `validation_step` の `self.log()` |
| `lr-SGD` | エポックごと | `LearningRateMonitor` が自動記録 |
| `val_predictions` | エポックごと（バッチ 0 のみ） | `validation_step` 内の `wandb.Image` |

---

## 推論時のロギング（`inference.py`）

推論でも独立した wandb Run を作成し、結果を記録します。

### `wandb.init()` による Run の開始

```python
run = wandb.init(
    project=args.wandb_project,
    job_type="inference",           # Run の種類を識別するタグ
    name=f"inference_{Path(args.checkpoint).stem}",
    dir=str(output_dir),
    config={
        "checkpoint": args.checkpoint,
        "score_thresh": args.score_thresh,
        "num_images": len(image_paths),
    },
)
```

- `job_type="inference"` : ダッシュボードでフィルタリングして学習 Run と見分けられる
- `config` : この推論に使ったパラメータを記録する。後から「どの閾値で推論したか」が追える
- 学習時は Lightning が `wandb.init()` を内部で呼ぶため不要でしたが、推論では明示的に呼びます

### `wandb.Image` による予測画像の記録

```python
wandb_img = wandb.Image(
    str(save_path),
    caption=f"{filename} — {n_detected} objects",
    boxes={
        "predictions": {
            "box_data": wandb_boxes,
            "class_labels": {i: c for i, c in enumerate(VOC_CLASSES)},
        }
    },
)
```

`wandb.Image` にバウンディングボックス情報を渡すと、ダッシュボード上で  
インタラクティブに箱を表示・非表示できるビジュアライゼーションになります。

`wandb_boxes` の形式：

```python
{
    "position": {"minX": ..., "minY": ..., "maxX": ..., "maxY": ...},
    "class_id": int,
    "scores": {"score": float},
}
```

---

## オフラインモード

ネットワーク接続なしに動作確認したい場合は `WANDB_MODE=offline` を指定します。

```bash
WANDB_MODE=offline PYTHONPATH=src python src/train.py
```

ローカルに `.wandb` ファイルが保存されます。後から以下でアップロードできます。

```bash
wandb sync outputs/train/2026-04-03/02-48-00/wandb/offline-run-*/
```

---

## ダッシュボードの活用

複数の Run を比較するには **Runs テーブル**が便利です。

- **X 軸を揃える** : `step` ではなく `epoch` で比較すると学習速度の違いを無視できる
- **フィルタ** : `job_type = "train"` など絞り込むと推論 Run が混ざらない
- **グルーピング** : `config.model.lr` でグループ化するとハイパーパラメータの効果を比較できる
