# syntax=docker/dockerfile:1
# CUDA 12.1 対応開発用イメージ
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# 環境変数の設定
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# ユーザー情報の引数 (デフォルト 1000:1000)
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=1000

# システムパッケージのインストールとクリーンアップ
# BuildKit のキャッシュマウントを使用して apt の再ダウンロードを抑制
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    git \
    curl \
    wget \
    vim \
    sudo \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# python3.11 をデフォルトの python として設定
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# ユーザーの作成 (ホストとの権限不一致を避けるため)
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# 仮想環境の作成と権限設定
RUN python3.11 -m venv $VIRTUAL_ENV \
    && chown -R $USERNAME:$USER_GID $VIRTUAL_ENV

WORKDIR /app

# 以降のコマンドは非ルートユーザーで実行
USER $USERNAME

# 1. 依存関係の定義ファイルを先にコピー
COPY --chown=$USERNAME:$USER_GID requirements-dev.txt setup.py setup.cfg README.md LICENSE ./

# 2. 依存関係をインストール (BuildKit キャッシュ使用)
RUN --mount=type=cache,target=/home/$USERNAME/.cache/pip,uid=$USER_UID,gid=$USER_GID \
    pip install --upgrade pip \
    && pip install -r requirements-dev.txt

# 3. 残りのソースコードをコピー
COPY --chown=$USERNAME:$USER_GID . .

# 4. パッケージを編集可能モードでインストール
RUN pip install -e .

# bash をデフォルトのコマンドにする
CMD ["/bin/bash"]
