# Pascal VOC 物体検出テンプレート

PyTorch Lightning + Weights & Biases (wandb) を使った Faster R-CNN 物体検出の研究用スターターテンプレートです。

## 技術スタック

| 項目 | 内容 |
|---|---|
| Python | 3.11 |
| PyTorch | 2.4.1+cu121 |
| PyTorch Lightning | 2.6.1 |
| モデル | Faster R-CNN (MobileNetV3-Large-320-FPN) |
| データセット | Pascal VOC 2012（初回実行時に自動ダウンロード・約 2 GB） |
| 実験管理 | Weights & Biases (wandb) |
| 設定管理 | Hydra |

---

## ディレクトリ構成

```text
.
├── configs/
│   └── train.yaml               # 学習の全パラメータをここで一元管理（Hydra）
├── docs/
│   ├── docker.md                # Docker・docker-compose の設定解説
│   ├── hydra.md                 # Hydra による設定管理・train.yaml 解説
│   ├── parser.md                # argparse による引数管理（inference.py 解説）
│   ├── pytorch_lightning.md     # LightningModule / DataModule / Trainer 解説
│   └── wandb.md                 # wandb セットアップ・ログ・推論時の記録
├── notebooks/
│   └── training_and_inference.ipynb  # 学習・推論の実行手順と証跡
├── src/
│   ├── detection_toolkit/
│   │   ├── datamodules/
│   │   │   └── voc_datamodule.py    # VOC のダウンロード・前処理・DataLoader
│   │   └── models/
│   │       └── detector.py          # LightningModule（モデル定義・学習/推論ロジック）
│   ├── train.py                 # 学習エントリーポイント
│   └── inference.py             # 推論エントリーポイント
├── data/                        # VOC データセット保存先（初回実行時に自動生成）
├── outputs/
│   ├── train/                   # 学習出力（チェックポイント・Hydra ログ・wandb キャッシュ）
│   └── runs/                    # 推論出力（バウンディングボックス付き結果画像）
├── Dockerfile
├── docker-compose.yml
└── requirements-dev.txt
```

> **全てのコマンドはコンテナ内の `/app` をカレントディレクトリとして実行します。**

---

## Step 1 — Docker 環境のビルドと起動

### 前提条件
- [Docker](https://docs.docker.com/engine/install/) がインストール済みであること
- GPU を使う場合は [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) もインストールすること（GPU がなくても CPU モードで動作します）

```bash
# 1. リポジトリをクローン
git clone <your-repo-url>
cd <your-repo>

# 2. イメージをビルド（初回は 5〜10 分かかります）
docker compose build

# 3. コンテナをバックグラウンドで起動
docker compose up -d

# 4. コンテナ内に入る
docker compose exec dev bash
```

> 以降のコマンドはすべて **コンテナ内の `/app`** で実行します。  
> Docker の詳細な設定については [docs/docker.md](docs/docker.md) を参照してください。

---

## Step 2 — Weights & Biases (wandb) のセットアップ

wandb は学習曲線・予測画像・ハイパーパラメータを自動でログする実験管理ツールです。

1. [https://wandb.ai/site](https://wandb.ai/site) でアカウントを作成し、API キーを取得
2. コンテナ内で以下を実行してログイン

```bash
wandb login
# プロンプトに API キーを貼り付けて Enter
# "Successfully logged in to Weights & Biases!" と表示されれば完了
```

> wandb の詳細な使い方は [docs/wandb.md](docs/wandb.md) を参照してください。

---

## Step 3 — 学習の実行

```bash
PYTHONPATH=src python src/train.py
```

VOC 2012 データセット（約 2 GB）は `data/` に初回のみ自動ダウンロードされます。

### パラメータの変更（Hydra）

yaml ファイルを書き換えなくても、コマンドラインで上書きできます。

```bash
# 学習率・バッチサイズを変更
PYTHONPATH=src python src/train.py model.lr=0.001 data.batch_size=4

# エポック数を増やす
PYTHONPATH=src python src/train.py trainer.max_epochs=20
```

> Hydra の詳細な使い方は [docs/hydra.md](docs/hydra.md) を参照してください。

### 学習済みモデルの保存先

```
outputs/train/YYYY-MM-DD/HH-MM-SS/
├── .hydra/                   ← この実行の設定スナップショット（再現性の担保）
├── checkpoints/
│   └── best-checkpoint.ckpt  ← val_loss が最良のエポックが自動保存される
└── wandb/                    ← wandb ローカルキャッシュ
```

---

## Step 4 — 推論の実行

学習が完了したら、保存された `.ckpt` を使って任意の画像に対して推論できます。

```bash
# ディレクトリ内の画像を一括処理
PYTHONPATH=src python src/inference.py \
  --checkpoint outputs/train/YYYY-MM-DD/HH-MM-SS/checkpoints/best-checkpoint.ckpt \
  --image_dir data/VOCdevkit/VOC2012/JPEGImages \
  --num_images 20

# 単一画像を処理する場合
PYTHONPATH=src python src/inference.py \
  --checkpoint outputs/train/YYYY-MM-DD/HH-MM-SS/checkpoints/best-checkpoint.ckpt \
  --image data/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg
```

結果画像（バウンディングボックス付き）は `outputs/runs/YYYY-MM-DD/HH-MM-SS/` に保存されます。

### 主なオプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--checkpoint` | （必須） | 使用する `.ckpt` ファイルのパス |
| `--image` | — | 単一画像のパス（`--image_dir` と排他） |
| `--image_dir` | — | 画像ディレクトリのパス（`--image` と排他） |
| `--num_images` | `20` | `--image_dir` 指定時に処理する枚数 |
| `--score_thresh` | `0.5` | 表示するバウンディングボックスのスコア閾値 |

---

## Notebook で手順を確認する

[notebooks/training_and_inference.ipynb](notebooks/training_and_inference.ipynb) に、上記の Step 1〜4 を Notebook 形式でまとめてあります。実行済みの出力（学習ログ・推論結果画像）が残っているため、手元で動かす前に全体の流れと結果を確認できます。

---

## 各ツールの詳細ドキュメント

| ドキュメント | 内容 |
|---|---|
| [docs/docker.md](docs/docker.md) | Dockerfile の各命令・docker-compose.yml の各設定を詳解。よくあるトラブル対処も記載 |
| [docs/hydra.md](docs/hydra.md) | `train.yaml` の各セクション解説、`@hydra.main`・`DictConfig`・`HydraConfig` の使い方、出力ディレクトリの仕組み |
| [docs/parser.md](docs/parser.md) | `argparse` の基本から `add_mutually_exclusive_group` まで `inference.py` の引数定義を詳解。Hydra との使い分けも説明 |
| [docs/pytorch_lightning.md](docs/pytorch_lightning.md) | `LightningModule`・`LightningDataModule`・`Trainer`・コールバックを実コードと照らして解説 |
| [docs/wandb.md](docs/wandb.md) | 基本概念の定義、セットアップ、学習・推論それぞれのロギング方法、オフラインモード |

---

## ライセンス

[MIT LICENSE](LICENSE)
