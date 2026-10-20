---
name: pr
description: "現在のブランチから、PR テンプレートに沿った GitHub PR を作成する"
argument-hint: "[関連Issue番号]"
---

# /pr

現在のブランチから GitHub PR を作成する。

## 手順

1. `git status` で未コミットの変更がないか確認する。あればユーザーに確認する
2. `git log main..HEAD --oneline` でコミット一覧を確認する
3. `git diff main...HEAD --stat` で変更ファイルを確認し、`data/`・`outputs/`・`workspace/`・重み・`.env` が含まれていないことを確かめる
4. `/dev-check` をまだ実行していなければ先に実行する
5. ブランチを push し、`.github/PULL_REQUEST_TEMPLATE.md` に沿って本文を作成する
6. 作成した PR の URL を報告する

## ルール

- タイトルは Conventional Commits 形式
- `main` へ直接 push しない
- **マージはユーザーが行う。エージェントはマージしない**
- Issue 対応なら `## 関連Issue` に番号を入れる（なければ「なし」）
- `## 動作確認` には実際に行ったものだけチェックを付ける
- 学習などの重い検証を省略した場合は `## 補足情報` に理由を書く

## ひな形

```bash
gh pr create --title "<type>: <summary>" --body "$(cat <<'EOF'
## 変更内容
- <変更点>

## 関連Issue
- #<番号>

## 動作確認
- [x] `python3 -m compileall src`
- [ ] 学習の疎通確認（`trainer.max_epochs=1`）
- [ ] 推論の疎通確認

## 補足情報
<重い検証を省略した理由や前提条件>
EOF
)"
```
