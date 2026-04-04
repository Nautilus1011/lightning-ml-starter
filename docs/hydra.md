# Hydra による設定管理

[Hydra](https://hydra.cc/) は Facebook Research が開発した設定管理ライブラリです。  
YAML ファイルでパラメータを一元管理しながら、コマンドラインから自由に上書きできる仕組みを提供します。

「実験ごとに `config.py` やスクリプトを書き換える」という非効率を解消し、  
**設定の変更履歴をコードと分離して管理できる**のが最大の利点です。

---

## Hydra を使う理由

| 課題 | Hydra による解決策 |
|---|---|
| 実験ごとにスクリプトを書き換える | YAML に外出しし、コマンドラインで上書き |
| 「あの時と同じ設定」が再現できない | 実行ごとに `.hydra/` フォルダへ設定スナップショットを自動保存 |
| 出力ファイルが混在する | 実行ごとにタイムスタンプ付きディレクトリを自動生成 |
| ハイパーパラメータの探索が大変 | WandB Sweep などと組み合わせてコマンド 1 行で変更できる |

---

## 設定ファイルの場所

```
configs/
└── train.yaml   # 学習パラメータをすべてここで管理
```

---

## `configs/train.yaml` の解説

### `# @package _global_`

```yaml
# @package _global_
```

Hydra の「パッケージ」宣言です。`_global_` を指定すると、このファイルの設定がルートスコープに展開されます。  
複数の YAML を階層化するときに名前空間を制御するための仕組みですが、単一ファイル構成なら `_global_` で問題ありません。

---

### `hydra:` — Hydra 自体の動作設定

```yaml
hydra:
  run:
    dir: outputs/train/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

Hydra フレームワーク自身の挙動を制御するセクションです。

- `run.dir` : スクリプト実行時の**作業ディレクトリ（出力先）**を指定する
- `${now:%Y-%m-%d}` : Hydra の組み込み変数。実行時刻をフォーマットして埋め込む  
  → 実行ごとに `outputs/train/2026-04-03/02-48-00/` のようなディレクトリが自動で作られる

この仕組みにより、複数実験の結果が上書きされることなく独立して保存されます。

---

### `logger:` — WandB ロガーの設定

```yaml
logger:
  project_name: "voc-object-detection"
  experiment_name: "fasterrcnn_mobilenet_v3_320"
```

WandB（実験管理ツール）に渡す識別情報です。  
Hydra 自体はこのセクションを直接解釈しません。`train.py` が `cfg.logger.project_name` として読み取り、`WandbLogger` に渡します。  

- `project_name` : WandB ダッシュボード上のプロジェクトの名前
- `experiment_name` : 個々の実験 run の名前。モデル名などを入れると後で見分けやすい

---

### `data:` — データ関連の設定

```yaml
data:
  data_dir: "${hydra:runtime.cwd}/data"
  batch_size: 8
  num_workers: 4
```

- `data_dir: "${hydra:runtime.cwd}/data"` : Hydra の実行時変数を使ったパス解決。  
  `hydra:runtime.cwd` はスクリプトを起動したディレクトリ（プロジェクトルート）に展開されるため、どのディレクトリから実行しても `data/` を正しく指せる
- `batch_size` : 1 ステップで処理するサンプル数。GPU メモリと学習速度のトレードオフ
- `num_workers` : DataLoader が使う並列プロセス数。`0` にするとメインプロセスのみ（Docker/Notebook 環境での安定動作向け）

---

### `model:` — モデルのハイパーパラメータ

```yaml
model:
  num_classes: 21
  lr: 0.005
  momentum: 0.9
  weight_decay: 0.0005
```

- `num_classes: 21` : VOC データセットの 20 クラス + 背景（class_id=0）で合計 21
- `lr` : 学習率。最も実験で変えることが多いパラメータ。コマンドラインで `model.lr=0.001` と上書きするのが典型的
- `momentum` : SGD オプティマイザの慣性項。勾配の方向を維持して収束を安定させる
- `weight_decay` : L2 正則化の係数。過学習を抑えるために重みが大きくなりすぎないようにペナルティを与える

---

### `trainer:` — Lightning Trainer の設定

```yaml
trainer:
  max_epochs: 10
  accelerator: "auto"
  devices: "auto"
  precision: "16-mixed"
  log_every_n_steps: 10
```

- `max_epochs` : 学習の最大エポック数（EarlyStopping により途中で終わることもある）
- `accelerator: "auto"` : GPU があれば `cuda`、なければ `cpu` を自動で選択する
- `devices: "auto"` : 利用可能なデバイス（GPU）の数を自動検出する
- `precision: "16-mixed"` : 混合精度学習（FP16 と FP32 を混用）。メモリを半減し、Tensor Core を使って高速化できる
- `log_every_n_steps` : 何ステップごとにメトリクスを記録するか

---

### `callbacks:` — コールバックの設定

```yaml
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

コールバックは学習ループの特定タイミングに自動で呼ばれる処理です。

**`early_stopping`**

| キー | 意味 |
|---|---|
| `monitor: "val_loss"` | 監視する指標。バリデーション損失が改善しなければ学習を止める |
| `patience: 3` | 改善がなくてもあと 3 エポック待ってから停止する |
| `mode: "min"` | 指標が「小さいほど良い」という意味（`val_loss` の場合は `min`） |

**`model_checkpoint`**

| キー | 意味 |
|---|---|
| `monitor: "val_loss"` | 保存判断に使う指標 |
| `save_top_k: 1` | 最も良いチェックポイントを 1 つだけ保存する（ストレージを節約） |
| `mode: "min"` | `val_loss` が最小のモデルを保存する |
| `filename: "best-checkpoint"` | 保存ファイル名（`.ckpt` が自動で付く） |

---

## `train.py` での使い方

### `@hydra.main` デコレータ

```python
# src/train.py
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def train(cfg: DictConfig):
    ...
```

`@hydra.main` を付けるだけで、関数実行前に Hydra が YAML を読み込み、`cfg` として渡してくれます。

| 引数 | 意味 |
|---|---|
| `version_base=None` | Hydra のバージョン互換性フラグ。`None` にすると現在のバージョンの動作に従う |
| `config_path="../configs"` | YAML ファイルがあるディレクトリ（`train.py` からの相対パス） |
| `config_name="train"` | 読み込む YAML のファイル名（拡張子 `.yaml` は省略） |

---

### `DictConfig` — 設定オブジェクトへのアクセス

```python
# ドット記法でネストしたキーにアクセスできる
dm = VOCDataModule(
    data_dir=cfg.data.data_dir,     # configs/train.yaml の data.data_dir
    batch_size=cfg.data.batch_size,
    num_workers=cfg.data.num_workers,
)
model = VOCDetector(
    num_classes=cfg.model.num_classes,
    lr=cfg.model.lr,
    momentum=cfg.model.momentum,
    weight_decay=cfg.model.weight_decay,
)
```

`DictConfig` は辞書のように見えますが、`.キー名` でアクセスできる点と、  
存在しないキーへのアクセス時にわかりやすいエラーが出る点が通常の `dict` より便利です。

---

### `OmegaConf.to_yaml(cfg)` — 設定の確認

```python
from omegaconf import OmegaConf
print(OmegaConf.to_yaml(cfg))
```

実行時に使われている設定を YAML 形式で出力します。コマンドラインで値を上書きしたときに、  
何が変わったかを確認するデバッグ手段として有効です。

---

### `HydraConfig.get()` — Hydra の実行時情報へのアクセス

```python
from hydra.core.hydra_config import HydraConfig
from pathlib import Path

run_dir = Path(HydraConfig.get().runtime.output_dir)
# → outputs/train/2026-04-03/02-48-00 のような絶対パス
```

`HydraConfig` には Hydra が内部で決定した値（出力ディレクトリのパスなど）が入っています。  
このリポジトリでは、チェックポイントの保存先を `run_dir / "checkpoints"` として動的に決めるために使っています。

---

## コマンドラインからパラメータを上書きする

YAML を書き換えなくても、実行時に `キー=値` の形式で任意の値を上書きできます。  
ネストしたキーはドット区切りで指定します。

```bash
# 学習率を変更
PYTHONPATH=src python src/train.py model.lr=0.001

# バッチサイズとエポック数を同時に変更
PYTHONPATH=src python src/train.py data.batch_size=4 trainer.max_epochs=5

# データ読み込みをシングルプロセスに（Docker / Notebook 環境で安定動作）
PYTHONPATH=src python src/train.py data.num_workers=0

# 混合精度を無効化（古い GPU 向け）
PYTHONPATH=src python src/train.py trainer.precision=32
```

---

## 出力ディレクトリの構造

実行ごとに以下のようなディレクトリが自動生成されます。

```
outputs/train/
└── 2026-04-03/
    └── 02-48-00/
        ├── .hydra/                  ← この実行の設定スナップショット（自動生成）
        │   ├── config.yaml          ← 最終的に使われた設定の全体像
        │   ├── hydra.yaml           ← Hydra 自体の設定
        │   └── overrides.yaml       ← コマンドラインで上書きした値だけ記録
        ├── checkpoints/
        │   └── best-checkpoint.ckpt
        └── train.log
```

**`.hydra/overrides.yaml` の例**（`model.lr=0.001` を指定した場合）

```yaml
- model.lr=0.001
```

`.hydra/` フォルダを見れば「あの実験はどんな設定で動かしたか」が即座にわかります。  
これが Hydra を使う最大のメリットである**実験の再現性**です。

---

## WandB Sweep との組み合わせ

Hydra のコマンドライン上書きは WandB Sweep と相性が良く、  
ハイパーパラメータ探索を自動化できます。詳細は [wandb_sweep.md](wandb_sweep.md) を参照してください。
