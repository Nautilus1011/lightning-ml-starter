# Experiment Report Rules — lightning-ml-starter

## 目的

ハイパーパラメータの比較やモデルの改良など、実験に取り組んだ際は内容を `EXPERIMENT.md` として残す。
使い方のドキュメント（`README.md` / `docs/`）ではなく、実験の記録として書く。

## 運用単位

- 1 ブランチ = 1 実験目的を想定する（ブランチ名は `research/<name>`）
- `EXPERIMENT.md` は `experiments/<name>/EXPERIMENT.md` に置く
- 実験の途中で目的が大きく変わった場合は、新しい実験ディレクトリを切る
  （1 つの `EXPERIMENT.md` に無関係な実験を詰め込まない）
- `/train-loop` で回した実験は、終了後に `workspace/train-loop/LOOP_LOG.md` の要点を `EXPERIMENT.md` にまとめる

## 構成

`EXPERIMENT.md` は以下の 4 セクションで構成する。

1. **目的 (Objective)** — 何を検証・改善したいか、背景
2. **実験条件 (Setup & Hyperparameters)** — 実行環境、モデル・パラメータ・設定
3. **実験結果 (Results)** — 主要メトリクス、ログ・出力の抜粋
4. **考察とネクストアクション (Discussion & Next Steps)** — 結果の考察、次にやること

## テンプレート

```markdown
# 🧪 [実験の短いタイトル]

## 📌 1. 目的 (Objective)
* 何を検証・改善したいのか、一言で記述。
* **背景**: 現状の課題や、前回の実験結果からの地続きの問い。

## ⚙️ 2. 実験条件 (Setup & Hyperparameters)
> **環境**: GPU: RTX 3060 / CUDA 12.1 / PyTorch 2.4.1 / Lightning 2.6.1 / Python 3.11

| カテゴリ | パラメータ名 | 設定値 | 意図・備考 |
| :--- | :--- | :--- | :--- |
| **Data** | `data.batch_size` / `data.num_workers` | `8` / `4` | ベースライン設定 |
| **Model** | `model.lr` / `model.momentum` | `0.005` / `0.9` | ベースライン設定 |
| **Trainer** | `trainer.max_epochs` / `trainer.precision` | `10` / `16-mixed` | |
| **Logger** | `logger.experiment_name` | `baseline` | W&B の run 名 |
| **Output** | Hydra の出力先 | `outputs/train/2026-01-01/12-00-00` | `.hydra/` に設定スナップショット |

## 📊 3. 実験結果 (Results)
* **主要メトリクス**: `val_loss 0.42`（ベースライン 0.48）/ 学習時間 `1h20m`
* **W&B**: run の URL
* **ログ/出力（一部抜粋）**:
  ```text
  Epoch 9: val_loss=0.420
  ```

## 💡 4. 考察とネクストアクション (Discussion & Next Steps)
### 🧐 考察 (Why?)
1. 結果に対する仮説や解釈。

### 🚀 ネクストアクション (Next)
* [ ] 次に試すこと。
```

## 注意事項

- 実行できなかった検証・省略した条件は隠さず、結果か考察のセクションに理由を明記する
- パラメータ表の値は、実際に使った設定（`.hydra/config.yaml` や override）と一致させる
- 学習済みモデル・生データ・大きな画像を `EXPERIMENT.md` に埋め込まない
- 乱数シード・データ分割・学習時間も記録し、再現可能な形で残す
