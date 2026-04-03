# Pascal VOC 物体検出テンプレート

PyTorch Lightning + Weights & Biases (wandb) を使った物体検出の研究用スターターテンプレートです。
研究室の後輩が「手動のログ管理を卒業し、WandB Sweep による自動パラメータ探索や Callback による効率化」を体験できることを目指しています。

---

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
├── src/
│   ├── configs/
│   │   └── config.yaml          # 学習の全パラメータをここで一元管理
│   ├── detection_toolkit/
│   │   ├── datamodules/
│   │   │   └── voc_datamodule.py  # VOC のダウンロード・前処理・DataLoader
│   │   └── models/
│   │       └── detector.py        # LightningModule（モデル定義・学習/推論ロジック）
│   ├── train.py                 # 学習エントリーポイント
│   └── inference.py             # 推論エントリーポイント
├── data/                        # VOC データセット保存先
├── outputs/
│   ├── checkpoints/             # 学習済みモデル (.ckpt)
│   ├── inference/               # 推論結果画像
│   ├── hydra/                   # Hydra 設定スナップショット・ログ
│   └── wandb/                   # wandb ローカルログ
├── scripts/
│   └── smoke_test.py            # データなしでパイプラインを動作確認するスクリプト
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

以降のコマンドはすべて **コンテナ内** で実行します。

---

## Step 2 — Weights & Biases (wandb) のセットアップ

wandb は学習曲線・バウンディングボックス付き予測画像・ハイパーパラメータを自動でログする実験管理ツールです。

### 2-1. アカウント作成

1. [https://wandb.ai/site](https://wandb.ai/site) にアクセスして **"Sign up"**
2. GitHub / Google アカウント連携、またはメールアドレスで登録
3. ログイン後: 右上アバター → **User Settings** → **API keys** → **"New key"** でキーをコピー

### 2-2. Docker 内での認証

**方法 A — `.env` ファイルで渡す（推奨）**

プロジェクトルートに `.env` を作成しておくと、`docker compose up` 時に自動で注入されます。

```bash
# プロジェクトルート（コンテナの外）で実行
echo "WANDB_API_KEY=ここにAPIキーを貼り付け" > .env
```

> `.env` は `.gitignore` に追加しておくこと（APIキーを git に含めないため）。

**方法 B — コンテナ内で対話ログイン**

```bash
wandb login
# プロンプトに API キーをペーストして Enter
# "Successfully logged in to Weights & Biases!" と表示されれば完了
```

---

## Step 3 — 学習の実行

```bash
# コンテナ内 /app で実行
cd /app
PYTHONPATH=src python src/train.py
```

学習が始まると以下のように表示されます。

```
GPU available: True (cuda), used: True
wandb: Syncing run fasterrcnn_mobilenet_v3_320
wandb: 🚀 View run at https://wandb.ai/<your-name>/voc-object-detection/runs/...
Epoch 0:  15%|██ | 109/715 [00:19<01:50,  5.48it/s, train_loss_step=1.010]
```

> VOC 2012 データセット（約 2 GB）は `data/` に初回のみ自動ダウンロードされます。

### パラメータをその場で変更する（Hydra）

`config.yaml` を書き換えなくても、コマンドラインで上書きできます。

```bash
# 学習率・バッチサイズを変更
PYTHONPATH=src python src/train.py model.lr=0.001 data.batch_size=4

# エポック数を増やす
PYTHONPATH=src python src/train.py trainer.max_epochs=20

# DataLoader のワーカー数を指定（shm が十分あれば高速化できる）
PYTHONPATH=src python src/train.py data.num_workers=4
```

### wandb なしでデバッグ実行

```bash
WANDB_MODE=offline PYTHONPATH=src python src/train.py
```

### 学習済みモデルの保存先

学習を実行するたびにタイムスタンプ付きのディレクトリが自動生成されます。

```
outputs/train/YYYY-MM-DD/HH-MM-SS/
├── .hydra/                        ← Hydra の設定スナップショット
├── checkpoints/
│   └── best-checkpoint.ckpt      ← val_loss が最良のエポックが自動保存される
├── train.log
└── wandb/                         ← wandb ローカルキャッシュ
```

`config.yaml` の `callbacks.model_checkpoint.filename` でファイル名を変更できます。

### wandb ダッシュボードで確認できる内容

[https://wandb.ai](https://wandb.ai) のダッシュボードに以下がリアルタイムで記録されます。

| ログ項目 | 説明 |
|---|---|
| `train_loss` / `val_loss` | 学習・検証損失のグラフ |
| `train_loss_classifier` など | 各損失成分の内訳 |
| `lr-SGD` | 学習率の変化（`LearningRateMonitor`） |
| バウンディングボックス付き予測画像 | 検証フェーズの最初のバッチ（スコア > 0.5）|
| EarlyStopping の停止エポック | `patience=3` で自動停止 |

---

## Step 4 — 推論の実行

学習が完了したら、保存された `.ckpt` ファイルを使って任意の画像に対して推論できます。
推論結果（バウンディングボックス付き画像）は `outputs/runs/YYYY-MM-DD/HH-MM-SS/` に保存され、wandb にも記録されます。

### 単一画像の推論

```bash
cd /app
PYTHONPATH=src python src/inference.py \
  --checkpoint outputs/train/YYYY-MM-DD/HH-MM-SS/checkpoints/best-checkpoint.ckpt \
  --image data/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg
```

### ディレクトリ内の画像を一括処理

```bash
cd /app
PYTHONPATH=src python src/inference.py \
  --checkpoint outputs/train/YYYY-MM-DD/HH-MM-SS/checkpoints/best-checkpoint.ckpt \
  --image_dir data/VOCdevkit/VOC2012/JPEGImages \
  --num_images 20
```

### 主なオプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--checkpoint` | （必須） | 使用する `.ckpt` ファイルのパス |
| `--image` | — | 単一画像のパス（`--image_dir` と排他） |
| `--image_dir` | — | 画像ディレクトリのパス（`--image` と排他） |
| `--num_images` | `20` | `--image_dir` 指定時に処理する枚数 |
| `--score_thresh` | `0.5` | 表示するバウンディングボックスのスコア閾値 |
| `--output_dir` | `outputs/runs/YYYY-MM-DD/HH-MM-SS` | 結果画像の保存先（自動生成） |

### 推論の出力ディレクトリ構造

```
outputs/runs/YYYY-MM-DD/HH-MM-SS/
├── 2007_000032_pred.jpg    ← バウンディングボックス付き画像
└── wandb/                  ← wandb ローカルキャッシュ
```

### 自分で学習したモデルを指定する

複数回学習を回した場合など、特定のチェックポイントを指定できます。

```bash
# 例: 別のチェックポイントを使いたい場合
PYTHONPATH=src python src/inference.py \
  --checkpoint outputs/train/YYYY-MM-DD/HH-MM-SS/checkpoints/best-checkpoint.ckpt \
  --image_dir data/VOCdevkit/VOC2012/JPEGImages \
  --num_images 50 \
  --score_thresh 0.6
```

### 推論ログの wandb での確認

推論 run は学習 run と **`job_type` で自動的に分離**されます。

| run の種別 | `job_type` | フィルタ方法 |
|---|---|---|
| 学習 | `train` | ダッシュボード → Filters → `job_type = train` |
| 推論 | `inference` | ダッシュボード → Filters → `job_type = inference` |

---

## Step 5 — WandB Sweep（自動ハイパーパラメータ探索）

複数の設定を自動で試して最良のパラメータを見つける機能です。

```bash
# 1. sweep.yaml を作成
cat > /app/src/sweep.yaml << 'EOF'
program: src/train.py
method: bayes
metric:
  name: val_loss
  goal: minimize
parameters:
  model.lr:
    distribution: log_uniform_values
    min: 1e-4
    max: 1e-1
  data.batch_size:
    values: [4, 8]
  model.weight_decay:
    values: [0.0001, 0.0005, 0.001]
EOF

# 2. Sweep を登録（sweep ID が発行される）
cd /app && PYTHONPATH=src wandb sweep src/sweep.yaml

# 3. エージェントを起動（自動で複数 run を実行）
PYTHONPATH=src wandb agent <your-entity>/voc-object-detection/<sweep-id>
```

---

## Step 6 — 動作確認スクリプト（スモークテスト）

VOC データをダウンロードせずに、合成データでパイプライン全体（学習 → チェックポイント保存 → 推論）を素早く確認できます。

```bash
cd /app
PYTHONPATH=src python scripts/smoke_test.py
```

成功すると以下のように表示されます。

```
=======================================================
全テスト PASSED — パイプライン正常動作を確認
=======================================================
保存されたチェックポイント: /app/outputs/smoke_test/smoke-best.ckpt
```

---

## 設定ファイル詳細 (`configs/train.yaml`)

```yaml
hydra:
  run:
    dir: outputs/hydra/${now:%Y-%m-%d}/${now:%H-%M-%S}  # Hydraログの保存先

logger:
  project_name: "voc-object-detection"   # wandb ダッシュボードのプロジェクト名
  experiment_name: "fasterrcnn_mobilenet_v3_320"

data:
  data_dir: "./data"        # VOC データセットのパス（/app からの相対パス）
  batch_size: 8             # RTX 3060 (12GB VRAM) でのデフォルト
  num_workers: 0            # Docker 環境では 0 を推奨（shm_size 拡張後は 4 以上可）

model:
  num_classes: 21           # VOC 20クラス + 背景 1クラス
  lr: 0.005                 # 学習率（Sweep で探索する主なパラメータ）
  momentum: 0.9
  weight_decay: 0.0005

trainer:
  max_epochs: 10
  accelerator: "auto"       # GPU があれば自動使用、なければ CPU
  devices: "auto"
  precision: "16-mixed"     # RTX の Tensor Core を活用（ResNet50比 1.5倍速）
  log_every_n_steps: 10

callbacks:
  early_stopping:
    monitor: "val_loss"
    patience: 3             # 3エポック改善がなければ自動停止
    mode: "min"
  model_checkpoint:
    monitor: "val_loss"
    save_top_k: 1
    mode: "min"
    filename: "best-checkpoint"   # → outputs/train/YYYY-MM-DD/HH-MM-SS/checkpoints/best-checkpoint.ckpt に保存
```

---

## トラブルシューティング

### `No space left on device` エラーが出る
Docker の共有メモリ (`/dev/shm`) が不足している。`docker-compose.yml` に `shm_size: '4gb'` が設定済みなので、**`docker compose build` し直してから起動**することで解決します。それまでは `data.num_workers=0` で回避できます。

```bash
PYTHONPATH=src python src/train.py data.num_workers=0
```

### `ModuleNotFoundError: No module named 'lightning'`
`requirements-dev.txt` に `lightning` パッケージが含まれています。イメージを再ビルドするか、コンテナ内で手動インストールしてください。

```bash
pip install lightning hydra-core hydra-colorlog omegaconf
```

---

## ライセンス

[MIT LICENSE](LICENSE)
