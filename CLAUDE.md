# CLAUDE.md — lightning-ml-starter

このファイルは Claude Code などのエージェントが、このリポジトリで迷わず作業するための実務ガイドです。

## プロジェクト概要

- 名称: `lightning-ml-starter`
- 目的: PyTorch Lightning / Hydra / W&B / Docker を使った機械学習開発の進め方を、研究室のメンバーに共有するためのスターターテンプレート
- 題材: Pascal VOC 2012 を使った Faster R-CNN (MobileNetV3-Large-320-FPN) の物体検出
- 利用者: 機械学習の開発環境に慣れていないメンバーも含む。**分かりやすさと再現性を最優先する**
- このリポジトリの `.claude/` と `CLAUDE.md` は、それ自体が「コーディングエージェントとの開発の進め方」の見本でもある。設定を変えたら `docs/claude_code.md` の解説も更新する

## ディレクトリの見方

- `src/train.py`: 学習エントリーポイント（Hydra）
- `src/inference.py`: 推論エントリーポイント（argparse）
- `src/detection_toolkit/models/`: LightningModule（モデル定義・学習/検証ロジック）
- `src/detection_toolkit/datamodules/`: LightningDataModule（VOC のダウンロード・前処理・DataLoader）
- `configs/train.yaml`: 学習の全パラメータ
- `docs/`: 各ツールの解説ドキュメント（初学者向け）
- `notebooks/`: 実行結果を残した手順書ノートブック
- `data/`, `outputs/`, `models/`: Git 管理外（`.gitkeep` のみ追跡）

## 開発環境

標準環境は Docker（`docker-compose.yml` の `dev` サービス）です。コマンドはすべてコンテナ内の `/app` で実行します。

```bash
docker compose build
docker compose up -d
docker compose exec dev bash
```

ホストから 1 コマンドだけ実行する場合:

```bash
docker compose exec dev bash -lc "<command>"
```

## 実行コマンド

```bash
# 学習（Hydra の override で設定を上書きできる）
PYTHONPATH=src python src/train.py
PYTHONPATH=src python src/train.py model.lr=0.001 trainer.max_epochs=20

# 推論
PYTHONPATH=src python src/inference.py \
  --checkpoint outputs/train/YYYY-MM-DD/HH-MM-SS/checkpoints/best-checkpoint.ckpt \
  --image_dir data/VOCdevkit/VOC2012/JPEGImages --num_images 20

# 最低限の検証
python3 -m compileall src
```

- 学習は重い処理です。初回は VOC 2012（約 2 GB）のダウンロードも走ります。データや GPU がない場合は無理に実行せず、`compileall` とコードレビューを最低線とします
- W&B にログインしていない環境では `WANDB_MODE=offline` を付けて実行します
- `tests/` はまだ空です。`tox` / `pre-commit` の設定は未整備のため、検証手段としては使いません

## 実装方針

1. 既存パターンを優先する。新しい抽象化は必要な時だけ入れる
2. モデルの変更は LightningModule、データの変更は DataModule に閉じ込める
3. 学習のパラメータは `configs/train.yaml` に集約し、コードに直書きしない
4. コードを変えたら、対応する `docs/` と `README.md` の記述も合わせて更新する
5. 研究コードでは再現性を優先し、パス・乱数シード・前提条件を明示する

## 実装完了の必須チェックリスト

**コミット前・PR 作成前に必ず全項目を確認すること。**

- [ ] 1 コミット = 1 つの論理的変更（原子コミット）
- [ ] Conventional Commits 形式（`feat:` / `fix:` / `docs:` / `chore:` 等）
- [ ] `git diff` が意図した差分だけになっている
- [ ] `data/`、`outputs/`、`workspace/`、学習済み重み、`.env` が含まれていない
- [ ] `python3 -m compileall src` が通る
- [ ] 実行できなかった検証があれば、PR の `## 補足情報` に理由を明記する
- [ ] PR 本文は `.github/PULL_REQUEST_TEMPLATE.md` に従っている
- [ ] **PR のマージはユーザーが実施する（エージェントはマージしない）**

## 参照ルール

- `.claude/rules/github-workflow.md`
- `.claude/rules/coding-standards.md`
- `.claude/rules/ml-research.md`
- `.claude/rules/experiment-report.md`
- `.claude/rules/project-guardrails.md`

## スキル

| スキル | 用途 |
|---|---|
| `/pr` | 現在のブランチから PR を作成する |
| `/issue` | テンプレートに沿って Issue を作成する |
| `/dev-check` | PR 前の動作確認を行う |
| `/nbexec` | コンテナ内でノートブックを実行して結果を保存する |
| `/train-loop` | Hydra override を切り替えながら学習→振り返りのループを回す |
| `/train-loop-status` | train-loop の進捗を確認する |
