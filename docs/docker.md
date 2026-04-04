# Docker 環境ガイド

Docker を使うと、OS やライブラリのバージョンに関係なく、誰でも同じ環境を再現できます。  
機械学習では「自分の PC では動いたのにサーバーでは動かない」という問題が頻発しますが、Docker はその解決策として広く使われています。

---

## Docker の基本概念

| 用語 | 意味 |
|---|---|
| **イメージ** | 環境の設計図。`Dockerfile` からビルドされる読み取り専用のスナップショット |
| **コンテナ** | イメージから起動した実行環境。プロセスが終われば中の変更は消える |
| **ボリューム** | ホストとコンテナ間でディレクトリを共有する仕組み。コンテナを消してもデータが残る |
| **レイヤー** | Dockerfile の各命令は「レイヤー」として積み重なる。変更がなければキャッシュが再利用される |

---

## 構成ファイル

| ファイル | 役割 |
|---|---|
| `Dockerfile` | イメージの定義。OS・ライブラリ・コードの積み重ね方を記述する |
| `docker-compose.yml` | コンテナの起動設定。ボリューム・GPU・共有メモリなどをまとめて管理する |
| `.env` | API キーなどの秘密情報（git 管理外）。`docker compose` が自動で読み込む |

---

## クイックスタート

```bash
# 1. Dockerfile をもとにイメージをビルド（初回は 5〜10 分）
docker compose build

# 2. コンテナをバックグラウンドで起動
docker compose up -d

# 3. 起動中のコンテナ内にシェルで入る
docker compose exec dev bash

# 4. 作業後にコンテナを停止・削除する
docker compose down
```

> 以降のコマンドはすべてコンテナ内の `/app` ディレクトリで実行します。

---

## Dockerfile の解説

`Dockerfile` はイメージを作るための命令書です。上から順に実行され、各命令が「レイヤー」として積み重なります。

### `# syntax=docker/dockerfile:1`

```dockerfile
# syntax=docker/dockerfile:1
```

BuildKit（Docker の高速ビルドエンジン）を有効にするための宣言です。  
この行を書くことで、後述の `--mount=type=cache` などの高度な機能が使えるようになります。

---

### `FROM` — ベースイメージの指定

```dockerfile
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel
```

すべての `Dockerfile` はどこかのイメージを土台（ベースイメージ）として始まります。  
ここでは PyTorch 公式イメージを使っているため、CUDA・cuDNN のセットアップを自分で行う必要がありません。

- `pytorch/pytorch` : Docker Hub 上の公式リポジトリ名
- `2.4.1` : PyTorch のバージョン
- `cuda12.1-cudnn9` : CUDA / cuDNN のバージョン（GPU 計算に必要なライブラリ）
- `devel` : コンパイルに必要なヘッダーや開発ツールを含むビルド向けバリアント

---

### `ENV` — 環境変数の設定

```dockerfile
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive
```

コンテナ内で常に有効な環境変数を定義します。イメージ全体に引き継がれます。

| 変数 | 効果 |
|---|---|
| `PYTHONDONTWRITEBYTECODE=1` | `.pyc` キャッシュファイルを生成しない。コンテナ内でのファイル散乱を防ぐ |
| `PYTHONUNBUFFERED=1` | Python の標準出力をバッファリングしない。ログがリアルタイムで `docker logs` に表示される |
| `DEBIAN_FRONTEND=noninteractive` | `apt-get` のインタラクティブなプロンプトを無効化する。ビルド中に止まらないようにするため |

---

### `ARG` — ビルド時引数の定義

```dockerfile
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=1000
```

`docker compose build` 時に外から渡せる変数です。`ENV` と異なり、ビルド完了後のコンテナ内では参照できません。

コンテナを非 root ユーザーで動かすためにユーザー名と UID/GID を定義しています。  
`docker-compose.yml` の `args:` セクションで値を上書きできます。

---

### `RUN` — コマンドの実行（システムパッケージのインストール）

```dockerfile
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git curl wget vim sudo openssh-client \
    libgl1-mesa-glx libglib2.0-0 build-essential \
    && rm -rf /var/lib/apt/lists/*
```

`RUN` はシェルコマンドを実行し、その結果を新しいレイヤーとして保存します。

- `--mount=type=cache` : ダウンロードしたパッケージをホスト側にキャッシュする BuildKit 機能。再ビルド時に `apt-get update` のダウンロードを省略できる
- `--no-install-recommends` : 推奨パッケージを含めない。イメージサイズを小さく保つため
- `&& rm -rf /var/lib/apt/lists/*` : apt のインデックスキャッシュを削除してイメージサイズを削減する。`--mount=type=cache` を使う場合は不要なこともあるが、明示的に残している

---

### `RUN` — 非 root ユーザーの作成

```dockerfile
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
```

コンテナを root で動かすとホスト側のマウントしたファイルがすべて root 所有になり、権限トラブルが起きます。  
ここでホストと同じ UID/GID を持つユーザーを作成することで、ホストで編集したファイルをコンテナ内でも問題なく読み書きできます。

- `groupadd` / `useradd` : Linux のグループ・ユーザー作成コマンド
- `-m` : ホームディレクトリ（`/home/devuser`）を同時に作成する
- `sudoers.d` : パスワードなしで `sudo` を使えるようにする設定

---

### `WORKDIR` — 作業ディレクトリの指定

```dockerfile
WORKDIR /app
```

以降の `RUN`・`COPY`・`CMD` 命令が実行される基準ディレクトリを設定します。  
`cd /app` と似ていますが、ディレクトリが存在しない場合は自動的に作成される点が異なります。

---

### `USER` — 実行ユーザーの切り替え

```dockerfile
USER $USERNAME
```

以降の命令を指定ユーザーとして実行します。  
この行以降は `devuser` として動作するため、作成されるファイルの所有者が `devuser` になります。

---

### `COPY` — ファイルのコピー

```dockerfile
COPY --chown=$USERNAME:$USER_GID requirements-dev.txt setup.py setup.cfg README.md LICENSE ./
```

ホスト（ビルドコンテキスト）からコンテナ内にファイルをコピーします。

- `--chown` : コピー先ファイルの所有者を指定する。これを付けないとファイルが root 所有になる
- ソースコード全体より先に依存定義ファイルだけをコピーしているのは **レイヤーキャッシュの最適化** のため。`requirements-dev.txt` が変わらない限り、次の `pip install` ステップのキャッシュが再利用される

---

### `RUN` — Python 依存関係のインストール

```dockerfile
RUN --mount=type=cache,target=/home/$USERNAME/.cache/pip,uid=$USER_UID,gid=$USER_GID \
    pip install --upgrade pip \
    && pip install -r requirements-dev.txt
```

- `--mount=type=cache` : pip のダウンロードキャッシュをホストに保持する。パッケージの再ダウンロードを防ぎビルドを高速化する
- `uid=$USER_UID,gid=$USER_GID` : `USER` 切り替え後でもキャッシュディレクトリに書き込める権限を与える

---

### `CMD` — コンテナ起動時のデフォルトコマンド

```dockerfile
CMD ["/bin/bash"]
```

`docker run` や `docker compose exec` でコマンドを指定しなかった場合に実行されるデフォルトコマンドです。  
`["/bin/bash"]` と配列形式（exec 形式）で書くと、シェルを介さず直接プロセスが起動するため、シグナル（Ctrl+C など）が正しく伝わります。

---

## docker-compose.yml の解説

`docker-compose.yml` は複数コンテナの構成や起動オプションを一ファイルにまとめる設定ファイルです。  
`docker run` に渡す長いオプションを管理しやすい形で記述できます。

### `services` — サービス（コンテナ）の定義

```yaml
services:
  dev:
    ...
```

`services` 以下に起動するコンテナをリストします。このリポジトリでは `dev` という 1 つのサービスだけを定義しています。  
`docker compose up` で `services` 配下のすべてのコンテナがまとめて起動します。

---

### `build` — イメージのビルド設定

```yaml
build:
  context: .
  args:
    USER_UID: 1000
    USER_GID: 1000
```

- `context: .` : ビルドコンテキストのパス。`Dockerfile` 内の `COPY` はここを起点にファイルを探す
- `args` : `Dockerfile` の `ARG` に渡す値。ここで UID/GID を自分の環境に合わせて変更できる  
  （`id -u` / `id -g` でホストの UID/GID を確認できます）

---

### `working_dir` — コンテナ内の作業ディレクトリ

```yaml
working_dir: /app
```

コンテナ起動直後のカレントディレクトリです。`Dockerfile` の `WORKDIR` と同じ値を指定することで一貫性を保ちます。

---

### `shm_size` — 共有メモリサイズ

```yaml
shm_size: '4gb'
```

`/dev/shm`（共有メモリ）の上限を設定します。

PyTorch の `DataLoader` は `num_workers > 0` のとき、ワーカープロセス間でデータを共有メモリ経由でやり取りします。  
Docker のデフォルトは 64MB と非常に小さく、そのままでは `num_workers` を増やすとクラッシュします。  
GPU 学習では `4gb` 程度を確保しておくのが安全です。

---

### `volumes` — ボリュームマウント

```yaml
volumes:
  - .:/app
```

`ホストのパス:コンテナのパス` の形式でディレクトリを共有します。

- `.:/app` : ホストのプロジェクトルート（`.`）をコンテナの `/app` にマウントする
- ホスト側でコードを編集すると即座にコンテナ内に反映される（再ビルド不要）
- コンテナを削除してもホスト側のファイルは消えない

---

### `tty` / `stdin_open` — インタラクティブ端末の有効化

```yaml
tty: true
stdin_open: true
```

- `tty: true` : 端末エミュレータ（疑似 TTY）を割り当てる。プロンプトの表示や色付き出力に必要
- `stdin_open: true` : 標準入力をオープンにする。`docker compose exec dev bash` でシェルに入ったときに入力できるようにするため

どちらも `docker run -it` の `-i`（stdin） と `-t`（tty）に相当します。

---

### `environment` — 環境変数の注入

```yaml
environment:
  - PYTHONUNBUFFERED=1
  - WANDB_API_KEY=${WANDB_API_KEY:-}
  - NVIDIA_VISIBLE_DEVICES=all
  - NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

コンテナ起動時に環境変数を設定します。

| 変数 | 説明 |
|---|---|
| `PYTHONUNBUFFERED=1` | Python ログをリアルタイム出力する（`Dockerfile` の `ENV` と同じ効果） |
| `WANDB_API_KEY=${WANDB_API_KEY:-}` | ホストの環境変数または `.env` ファイルから `WANDB_API_KEY` を取り込む。未設定のときは空文字になる（`:-` は Bash のデフォルト値構文） |
| `NVIDIA_VISIBLE_DEVICES=all` | コンテナから見える GPU を指定する。`all` ですべての GPU を公開する |
| `NVIDIA_DRIVER_CAPABILITIES=compute,utility` | コンテナに公開する NVIDIA ドライバ機能を制限する。`compute` は CUDA 計算、`utility` は `nvidia-smi` などのユーティリティ |

---

### `deploy.resources.reservations.devices` — GPU へのアクセス

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

Docker Compose v2 で GPU を使うための公式の記述方法です（`docker run --gpus all` に相当）。

- `driver: nvidia` : NVIDIA Container Toolkit を使って GPU へアクセスする
- `count: all` : 利用可能なすべての GPU をコンテナに割り当てる（`1` や `2` のように数も指定できる）
- `capabilities: [gpu]` : GPU デバイスとして認識させる

> GPU がない環境でもコンテナは起動し、CPU モードで動作します。

---

## wandb API キーの設定

プロジェクトルートに `.env` ファイルを作成しておくと `docker compose up` 時に自動で読み込まれます。

```bash
# ホスト側で実行
echo "WANDB_API_KEY=your_api_key_here" > .env
```

> `.env` は `.gitignore` に追加し、API キーを git に含めないようにしてください。

---

## よくあるトラブル

### `No space left on device` / DataLoader がクラッシュする

共有メモリ不足の可能性があります。`shm_size` を増やすか、`num_workers=0` で一時回避してください。

```bash
PYTHONPATH=src python src/train.py data.num_workers=0
```

### GPU が認識されない

[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) がホストにインストールされているか確認してください。

```bash
# ホスト側で確認
nvidia-smi

# コンテナ内で確認
python -c "import torch; print(torch.cuda.is_available())"
```

### ファイルの権限エラー（Permission denied）

`docker-compose.yml` の `args` の `USER_UID` / `USER_GID` をホストのユーザー ID に合わせてください。

```bash
# ホスト側で自分の UID/GID を確認
id -u && id -g
```
