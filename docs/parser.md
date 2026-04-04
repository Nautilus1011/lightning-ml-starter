# argparse による引数管理

`argparse` は Python 標準ライブラリのコマンドライン引数パーサーです。  
このリポジトリでは、`src/inference.py`（推論スクリプト）の実行オプションを管理するために使っています。

Hydra は設定ファイルを中心とした学習向きのツールですが、  
**推論のように「毎回チェックポイントや画像パスが変わる」用途には `argparse` がシンプルで明快**です。

---

## argparse を使う理由

| 比較 | argparse | Hydra |
|---|---|---|
| 設定ファイル | 不要 | YAML が必要 |
| 引数の定義 | コードに直接書く | YAML に書く |
| 向いている用途 | 推論・変換スクリプトなど都度パスが変わるもの | 学習など実験管理が必要なもの |
| 再現性の記録 | 自前で管理 | `.hydra/` に自動保存 |
| 標準ライブラリ | ◎（インストール不要） | 別途インストールが必要 |

---

## 基本的な使い方

```python
import argparse

parser = argparse.ArgumentParser(description="スクリプトの説明文")
parser.add_argument("--オプション名", type=型, default=デフォルト値, help="説明")
args = parser.parse_args()

# args.オプション名 でアクセス
print(args.オプション名)
```

`--` を付けた引数を「オプション引数」といい、省略可能です。  
`--` を付けない引数を「位置引数」といい、順序が固定で省略不可です。  
このリポジトリの `inference.py` ではすべてオプション引数を使っています。

---

## `inference.py` の引数定義 — 詳細解説

### `ArgumentParser` — パーサーの生成

```python
parser = argparse.ArgumentParser(description="VOC 物体検出 推論スクリプト")
```

パーサーオブジェクトを作成します。`description` に書いた文字列は `--help` を実行したときに表示されます。

```bash
# --help の出力例
PYTHONPATH=src python src/inference.py --help
```

---

### `--checkpoint` — 必須引数

```python
parser.add_argument("--checkpoint", type=str, required=True,
                    help="学習済みチェックポイントのパス (.ckpt)")
```

- `type=str` : 受け取った文字列をそのまま `str` として扱う
- `required=True` : この引数が省略された場合、エラーメッセージを出して終了する  
  → チェックポイントは推論に必須なので、省略できないように強制している

```bash
# --checkpoint を省略するとエラーになる
PYTHONPATH=src python src/inference.py --image foo.jpg
# error: the following arguments are required: --checkpoint
```

---

### `add_mutually_exclusive_group` — 排他グループ

```python
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--image",     type=str, help="推論する画像のパス（単体）")
group.add_argument("--image_dir", type=str, help="画像が入ったディレクトリのパス（一括処理）")
```

**排他グループ**は「どちらか一方だけ指定できる」制約をつける仕組みです。

- `required=True` をグループに指定 → `--image` か `--image_dir` のどちらか一方は必ず指定しなければならない
- 両方同時に指定するとエラーになる
- どちらも指定しないとエラーになる

```bash
# 正しい使い方（単体画像）
python src/inference.py --checkpoint model.ckpt --image photo.jpg

# 正しい使い方（ディレクトリ一括）
python src/inference.py --checkpoint model.ckpt --image_dir data/images/

# 両方指定するとエラー
python src/inference.py --checkpoint model.ckpt --image photo.jpg --image_dir data/
# error: argument --image_dir: not allowed with argument --image
```

---

### `--num_images` — デフォルト値付き引数

```python
parser.add_argument("--num_images", type=int, default=20,
                    help="--image_dir 指定時に処理する画像数 (default: 20)")
```

- `type=int` : 文字列として受け取った値を `int` に自動変換する。変換できない場合はエラー
- `default=20` : 指定しなかったときの値。`args.num_images` は `20` になる

`type` を指定しないとすべて `str` になります。数値比較や演算をする場合は明示的に型を指定することが重要です。

---

### `--score_thresh` — float 型引数

```python
parser.add_argument("--score_thresh", type=float, default=0.5,
                    help="表示する予測のスコア閾値 (default: 0.5)")
```

- `type=float` : `"0.7"` のような文字列を `0.7` の `float` に変換する
- 物体検出では信頼スコアが閾値以上の予測のみ表示するのが一般的  
  高くしすぎると何も表示されず、低くしすぎると誤検出が増える

---

### `--output_dir` — 動的デフォルト値

```python
_now = datetime.now()
default_output_dir = f"outputs/runs/{_now.strftime('%Y-%m-%d/%H-%M-%S')}"

parser.add_argument("--output_dir", type=str, default=default_output_dir,
                    help=f"結果の保存先 (default: {default_output_dir})")
```

デフォルト値をスクリプト起動時刻から動的に生成しています。  
これにより、`--output_dir` を省略しても実行ごとに別ディレクトリに結果が保存されます。

> Hydra では `${now:%Y-%m-%d}` で同じことを YAML 内で実現していますが、  
> argparse では Python コードで計算してデフォルト値として渡す方法が自然です。

---

### `--wandb_project`

```python
parser.add_argument("--wandb_project", type=str, default="voc-object-detection",
                    help="wandb プロジェクト名")
```

デフォルト値を持つ省略可能な引数です。  
省略した場合は `"voc-object-detection"` が使われ、別プロジェクトに記録したいときだけ指定します。

---

### `parse_args()` — 引数の解析

```python
args = parser.parse_args()
```

実際にコマンドライン引数を解析してオブジェクトを返します。  
この時点で型変換・必須チェック・排他チェックがすべて行われ、問題があればエラーを出して終了します。  

解析後は `args.引数名` でアクセスします（`--` は除かれ、`-` は `_` に変換されます）。

```python
# --checkpoint → args.checkpoint
# --image_dir  → args.image_dir
# --score_thresh → args.score_thresh
print(args.checkpoint)   # "outputs/train/.../best-checkpoint.ckpt"
print(args.score_thresh) # 0.5  （floatとして取得できる）
```

---

## 実行例まとめ

```bash
# 単体画像の推論
PYTHONPATH=src python src/inference.py \
  --checkpoint outputs/train/2026-04-03/02-48-00/checkpoints/best-checkpoint.ckpt \
  --image data/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg

# ディレクトリ内の 50 枚を、スコア閾値 0.7 で推論
PYTHONPATH=src python src/inference.py \
  --checkpoint outputs/train/2026-04-03/02-48-00/checkpoints/best-checkpoint.ckpt \
  --image_dir data/VOCdevkit/VOC2012/JPEGImages \
  --num_images 50 \
  --score_thresh 0.7

# 出力先を明示的に指定
PYTHONPATH=src python src/inference.py \
  --checkpoint outputs/checkpoints/best-checkpoint.ckpt \
  --image_dir data/VOCdevkit/VOC2012/JPEGImages \
  --output_dir outputs/inference/my_experiment
```

---

## Hydra との比較まとめ

```
学習スクリプト（train.py）
  → Hydra を使用
  → YAML で設定を一元管理
  → 実験ごとに .hydra/ に設定スナップショットを自動保存
  → コマンドライン上書き: model.lr=0.001

推論スクリプト（inference.py）
  → argparse を使用
  → コードに引数定義を直接記述
  → 実行のたびに --checkpoint や --image を指定する
  → コマンドライン引数: --score_thresh 0.7
```

どちらが優れているというわけではなく、用途に応じて使い分けることが大切です。
