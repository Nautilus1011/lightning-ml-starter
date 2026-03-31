# Python Machine Learning Template

このリポジトリは、Docker (CUDA 12.1対応) を使用した Python 機械学習プロジェクトのためのテンプレートリポジトリです。
研究開発やプロトタイピングに最適なディレクトリ構成と環境構築設定があらかじめ用意されています。

## 特徴

- **GPU 対応**: NVIDIA Docker (CUDA 12.1, cuDNN 8) をベースにした開発環境。
- **ML 標準構成**: 機械学習プロジェクトで一般的に使われるディレクトリ構造を網羅。
- **パッケージ管理**: `setup.cfg` と `requirements-dev.txt` による柔軟な依存関係管理。
- **VSCode 連携**: `.vscode` 設定が含まれており、コンテナ内での開発がスムーズ。

## ディレクトリ構成

```text
.
├── configs/            # 実験の設定ファイル（YAML, JSONなど）
├── data/               # データセット（raw, processed, external）
├── docs/               # プロジェクトのドキュメント、論文、メモ
├── models/             # 学習済みモデルやチェックポイントの保存先
├── notebooks/          # Jupyter Notebooks (EDAやプロトタイピング)
├── outputs/            # 実行結果、ログ、図、グラフ
├── scripts/            # 一回限りの実行スクリプトやユーティリティ
├── src/                # メインのソースコード
│   └── my_ml_project/  # プロジェクト固有のモジュール
├── tests/              # ユニットテスト
├── Dockerfile          # 開発環境用 Docker イメージの定義
├── docker-compose.yml  # コンテナ実行設定 (GPU割り当て等)
├── requirements-dev.txt # 開発用パッケージ一覧
├── setup.cfg           # パッケージのメタデータと依存関係
└── setup.py            # パッケージインストール用
```

## セットアップ手順

このテンプレートから新しいプロジェクトを作成した後の手順です。

### 1. 前提条件

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) または Docker Engine
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (GPUを使用する場合)
- [VSCode](https://code.visualstudio.com/) (推奨) + [Dev Containers 拡張機能](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### 2. 環境構築 (Docker)

#### VSCode を使用する場合 (推奨)
1. VSCode でプロジェクトのルートディレクトリを開きます。
2. 左下の緑色のアイコン（または `F1` キーで `Dev Containers: Reopen in Container` を選択）をクリックします。
3. 自動的にイメージのビルドとコンテナの起動が行われ、コンテナ内の開発環境に接続されます。

#### コマンドラインを使用する場合
コンテナをバックグラウンドで起動し、シェルに入る場合は以下を実行します：

```bash
docker-compose up -d
docker-compose exec dev bash
```

### 3. パッケージのカスタマイズ

1. `setup.cfg` の `metadata` セクション（`name`, `author` など）を自分のプロジェクトに合わせて変更します。
2. `src/my_ml_project` ディレクトリを自分のプロジェクト名に変更します。
3. 必要に応じて `requirements-dev.txt` にライブラリを追加します。

## 開発ガイドライン

- **ソースコード**: 共通のロジックやモデル定義は `src/` 内に記述し、パッケージとして管理します。
- **ノートブック**: 実験や分析は `notebooks/` で行い、完成したコードは `src/` へリファクタリングして移行します。
- **実験管理**: ハイパーパラメータなどは `configs/` 内のファイルで管理することを推奨します。

---

## ライセンス

[MIT LICENSE](LICENSE)
