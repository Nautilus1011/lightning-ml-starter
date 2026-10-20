# Python Coding Standards — lightning-ml-starter

## 基本方針

- インデントは 4 スペース（Tab 禁止）
- 命名規則:
  - クラス名: `PascalCase`
  - 関数・変数・モジュール名: `snake_case`
  - 定数: `UPPER_SNAKE_CASE`
- 既存コードの文体（日本語の docstring・コメント）に合わせる
- このリポジトリは学習教材も兼ねる。初学者がつまずきやすい処理（Faster R-CNN の train/eval モードの切り替えなど）には、なぜそうするのかをコメントで残す。コードを読めば分かる WHAT のコメントは書かない
- 新しい公開関数・クラスには短い docstring を付ける

---

## 型ヒント

- 公開関数には引数と戻り値の型ヒントを付ける
- 複雑なデータ構造には `TypedDict` や `dataclass` を活用する

```python
def build_targets(boxes: list[list[float]], labels: list[int]) -> dict[str, torch.Tensor]:
    """VOC のアノテーションを Faster R-CNN の target 形式に変換する。"""
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }
```

---

## Lightning の責務分担

| 置き場所 | 書くもの |
|---|---|
| `LightningModule` | モデル定義、`training_step` / `validation_step`、optimizer と scheduler |
| `LightningDataModule` | ダウンロード、前処理、Dataset、DataLoader、`collate_fn` |
| `src/train.py` | 設定の読み込み、Logger・Callback・Trainer の組み立てのみ |

- `train.py` に学習ロジックを書かない
- ログに出す値は `self.log` / `self.log_dict` で記録し、名前は `train_*` / `val_*` に揃える

---

## 設定とパス

- 学習のパラメータは `configs/train.yaml` に置き、`cfg.<section>.<key>` で参照する
- 新しい設定項目を追加したら `docs/hydra.md` の解説も更新する
- 絶対パスを書かない。データの場所は `${hydra:runtime.cwd}` を基準にする

---

## 依存関係

- `requirements-dev.txt` を唯一の入口とする
- パッケージを追加したら Docker イメージを再ビルドして動作を確認する

---

## テストと検証

| 検証種別 | 手段 | 条件 |
|---|---|---|
| 構文チェック | `python3 -m compileall src` | 常に実施 |
| 学習の疎通確認 | `trainer.max_epochs=1` で学習 | データと計算資源がある場合 |
| 推論の疎通確認 | `src/inference.py --image <1枚>` | チェックポイントがある場合 |

- 実行できなかった検証は隠さず、PR の `## 補足情報` に理由を明記する
