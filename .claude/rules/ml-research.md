# ML Research Rules — lightning-ml-starter

## データ管理

- `data/` 配下のデータセットは Git に含めない
- `outputs/` の実行結果（チェックポイント・Hydra ログ・wandb キャッシュ・推論画像）はコミットしない
- 学習済み重み（`.ckpt` / `.pt` / `.onnx` 等）を新規に Git 管理に追加しない
- クラス定義や前処理を変えた場合は、変更内容を `docs/` か Issue に残す

---

## 学習と評価

- 比較実験では、ベースライン条件を固定してから差分を 1 つずつ増やす
- 学習結果を報告する時は、何を固定して何を変えたかを明示する
- `val_loss` だけで判断せず、W&B の予測画像や推論結果の目視確認も合わせて行う
- 学習データと検証データの評価を混在させない
- 小規模な設定で end-to-end の動作を確認してから本番の学習を回す

---

## 設定管理（Hydra）

- 学習スクリプトでは `@hydra.main` を使い、`configs/` の YAML から設定を読み込む
- パス・閾値・ハイパーパラメータをコード内にハードコードしない
- 実験はコマンドラインの override で行い、YAML の既定値は安易に書き換えない

```bash
PYTHONPATH=src python src/train.py model.lr=0.001 data.batch_size=4 trainer.max_epochs=20
```

- 推論スクリプト（`src/inference.py`）は argparse を使う。使い分けの理由は `docs/parser.md` を参照

---

## 実験管理（W&B）

- run 名（`logger.experiment_name`）は、何を試したか分かる名前にする
- 結果を報告する時は、run 名だけでなく Hydra の出力ディレクトリも合わせて示す
- オフラインで実行した run は、後から `wandb sync` で同期できる

---

## ノートブック

- `notebooks/training_and_inference.ipynb` は手順書兼証跡。セルを上から順に実行して再現できる状態に保つ
- 探索用のノートブックで書いたロジックは、再利用するなら `src/` に移植する
- 大きなデータや学習済みモデルをノートブック内に直接保存しない

---

## 再現性

- 乱数シード、データの分割、使用した GPU と学習時間を記録する
- 実行ごとの設定は Hydra が `outputs/train/<日付>/<時刻>/.hydra/` に残す。報告ではこのパスを示す
- 重い実験を省略した場合も、再現に必要な前提（データ、環境、パラメータ）を書き残す
