---
name: issue
description: "Issue テンプレートに沿って GitHub Issue を作成する"
argument-hint: "<作りたい Issue の内容>"
---

# /issue

GitHub Issue を作成する。内容: "$ARGUMENTS"

## 手順

1. 機能追加・改善か、バグ報告かを判断する
2. `gh issue list --state open` で重複がないか確認する
3. 下のひな形を埋め、作成前にタイトルと本文をユーザーに見せて確認を取る
4. 作成した Issue の URL を報告する

## ルール

- `.github/ISSUE_TEMPLATE/` のテンプレートに従う
  - 機能追加: `feature_request.md`（ラベル `enhancement`）
  - バグ: `bug_report.md`（ラベル `bug`）
- タイトルは、一覧で見た時に判別できる粒度にする
- 大きすぎる場合は「実装」「実験」「ドキュメント」に分けて複数の Issue にする
- 情報が足りない項目は推測で埋めず、ユーザーに確認する

## 機能追加のひな形

```bash
gh issue create --label enhancement --title "<タイトル>" --body "$(cat <<'EOF'
## 概要

<追加または変更する機能の説明>

## 背景

<なぜ必要か>

## スケジュール

<確認時期や目安>

## 補足事項 (optional)

<補足>
EOF
)"
```

## バグ報告のひな形

```bash
gh issue create --label bug --title "<タイトル>" --body "$(cat <<'EOF'
## 概要
<バグの要約>

## 再現手順
1.
2.
3.

## 期待される動作
<期待値>

## 実際の動作
<実際の挙動、エラーメッセージ>

## 環境情報
- OS:
- Python version:
- その他 (GPU, CUDA など):

## 補足事項 (optional)

<補足>
EOF
)"
```
