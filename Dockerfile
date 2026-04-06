# syntax=docker/dockerfile:1
# 研究用：CPU/GPU 両対応イメージ
# PyTorch 公式イメージベース
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

# 環境変数の設定
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# ユーザー情報の引数 (デフォルト 1000:1000)
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=1000

# システムパッケージのインストール
# pytorch/pytorch イメージは Ubuntu ベース
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    vim \
    sudo \
    openssh-client \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ユーザーの作成
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

WORKDIR /app

USER $USERNAME

ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

# 依存関係のコピーとインストール
COPY --chown=$USERNAME:$USER_GID requirements-dev.txt setup.py setup.cfg README.md LICENSE ./

# その他の依存関係（PyTorch Lightningなど）をインストール
# --extra-index-url は requirements-dev.txt 内で管理
RUN --mount=type=cache,target=/home/$USERNAME/.cache/pip,uid=$USER_UID,gid=$USER_GID \
    pip install --upgrade pip \
    && pip install -r requirements-dev.txt

# ソースコードのコピー
COPY --chown=$USERNAME:$USER_GID . .

# 編集可能モードでプロジェクトをインストール
RUN pip install -e .

CMD ["/bin/bash"]
