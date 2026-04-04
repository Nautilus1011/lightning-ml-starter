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
| `val_mAP` | エポックごと | `on_validation_epoch_end` の `self.log()` |
| `val_mAP_50` | エポックごと | `on_validation_epoch_end` の `self.log()` |
| `val_mAP_75` | エポックごと | `on_validation_epoch_end` の `self.log()` |
| `lr-SGD` | エポックごと | `LearningRateMonitor` が自動記録 |
| `val_predictions` | エポックごと（バッチ 0 のみ） | `validation_step` 内の `wandb.Image` |

---

## mAP（mean Average Precision）

物体検出の標準的な評価指標です。`val_loss` と異なり、実際に検出できているかを直接測れます。

### 3 つの指標

| 指標 | 意味 | 基準 |
|---|---|---|
| `val_mAP` | mAP @ IoU=0.50:0.95 | COCO コンペ標準。厳しめの指標 |
| `val_mAP_50` | mAP @ IoU=0.50 | Pascal VOC 標準。ボックスが大まか一致すれば正解 |
| `val_mAP_75` | mAP @ IoU=0.75 | 精度を重視する場合の中間指標 |

**IoU（Intersection over Union）** はバウンディングボックスの重なり率です。  
IoU = 0.50 なら「予測と正解が 50% 以上重なれば正解」として TP を数えます。

### 実装のポイント

`torchmetrics` の `MeanAveragePrecision` を `__init__` で初期化し、  
`validation_step` で `update()`、エポック末の `on_validation_epoch_end` で `compute()` するパターンです。

```python
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# __init__ で初期化
self.val_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

# validation_step で蓄積（全バッチ分を蓄積してからエポック末に計算）
self.val_map.update(preds, targets)

# on_validation_epoch_end でエポック mAP を計算・ログ
map_metrics = self.val_map.compute()
self.log("val_mAP", map_metrics["map"], prog_bar=True)
self.val_map.reset()  # 次のエポックのためにリセット
```

> `val_loss` ではなく `val_mAP` をモデルの選択基準にすることで、  
> 実際の検出性能が最も高いチェックポイントを保存できます。  
> `configs/train.yaml` の `model_checkpoint.monitor` と `early_stopping.monitor` は  
> 既に `val_mAP`（`mode: max`）に設定済みです。

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

---

## WandB Sweep — ハイパーパラメータ自動探索

Sweep はハイパーパラメータの探索を自動化する機能です。  
`lr`, `weight_decay`, `batch_size` などの候補を指定すると、  
复数の Run を自動実行して最も良い組み合わせを見つけてくれます。

### 設定ファイル

探索の設定は `configs/sweep.yaml` で管理します。

```yaml
method: bayes   # ベイズ最適化（random / grid より効率的）

metric:
  name: val_mAP  # 最大化したい指標
  goal: maximize

parameters:
  model.lr:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.05
  data.batch_size:
    values: [4, 8]
```

**`method` の選択肢:**

| method | 特徴 | 使い分け |
|---|---|---|
| `bayes` | 過去の結果から次の試行点を予測 | パラメータが 5 個以下・試行回数を抑えたいとき（推奨） |
| `random` | ランダムサンプリング | 素早く傾向を掴みたいとき |
| `grid` | 全組み合わせを試す | パラメータが少なく完全探索したいとき |

### 実行手順

```bash
# 1. スイープを登録（プロジェクトに Sweep が作成される）
wandb sweep configs/sweep.yaml
# → "Created sweep with ID: abc123" のように表示される

# 2. エージェントを起動（自動で学習を繰り返す）
wandb agent <entity>/<project>/<sweep-id>
```

`wandb agent` を複数のターミナルや GPU マシンで同時に起動すると、  
並列で探索を進めることができます。

### Hydra との連携

このリポジトリでは `configs/sweep.yaml` の `command` セクションに  
`${args_no_hyphens}` を指定しています。

```yaml
command:
  - ${env}
  - python
  - ${program}
  - ${args_no_hyphens}   # "model.lr=0.001" 形式で渡す（Hydra の上書き構文）
```

これにより wandb Sweep が `model.lr=0.001` のように Hydra 形式で引数を渡すため、  
`train.py` の `@hydra.main` がそのまま上書き値として受け取れます。  
`train.py` のコードを一切変更せずに Sweep を使えるのがポイントです。

### ダッシュボードの Sweep ビュー

wandb のプロジェクトページに **Sweeps** タブが追加され、以下が確認できます。

- **Parameter importance** : どのパラメータが `val_mAP` に最も影響するか
- **Parallel coordinates plot** : パラメータの組み合わせと結果の可視化
- **Best run** : 最も高い `val_mAP` を記録した Run とそのパラメータ
