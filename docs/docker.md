# Docker 環境ガイド

このリポジトリは Docker コンテナ上で動作します。GPU の有無に関わらず同じ環境を再現できます。

---

## 構成ファイル

| ファイル | 役割 |
|---|---|
| `Dockerfile` | イメージ定義（PyTorch 公式イメージベース） |
| `docker-compose.yml` | コンテナ起動設定（ボリューム・GPU・共有メモリ） |
| `.env` | API キーなどの秘密情報（git 管理外） |

---

## クイックスタート

```bash
# 1. イメージをビルド（初回は 5〜10 分）
docker compose build

# 2. コンテナをバックグラウンドで起動
docker compose up -d

# 3. コンテナ内に入る
docker compose exec dev bash

# 4. 作業後にコンテナを止める
docker compose down
```

> 以降のコマンドはすべてコンテナ内の `/app` で実行します。

---

## Dockerfile の概要

```dockerfile
# PyTorch 公式イメージ（CUDA 12.1 + cuDNN 9）
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

# 非 root ユーザーで動かす（UID/GID はビルド引数で変更可）
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=1000
```

- ベースに `pytorch/pytorch` 公式イメージを使うことで、CUDA / cuDNN のセットアップが不要
- 非 root ユーザー `devuser` でコンテナを動かすため、ホスト側のファイルの権限トラブルが起きにくい

---

## docker-compose.yml の重要な設定

```yaml
services:
  dev:
    shm_size: '4gb'          # DataLoader の num_workers 使用時に必要
    volumes:
      - .:/app               # ホストのプロジェクトルートをコンテナの /app にマウント
    environment:
      - WANDB_API_KEY=${WANDB_API_KEY:-}   # .env から自動注入
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

**ポイント**

- `volumes: .:/app` により、ホストで編集したコードがコンテナにリアルタイムで反映される
- `shm_size: '4gb'` は PyTorch DataLoader が共有メモリを使うために必要（不足すると `num_workers > 0` でクラッシュする）
- GPU がない環境でもコンテナは起動し、CPU モードで動作する

---

## wandb API キーの渡し方（推奨）

プロジェクトルートに `.env` ファイルを作成しておくと、`docker compose up` 時に自動で注入されます。

```bash
# コンテナの外（ホスト側）で実行
echo "WANDB_API_KEY=your_api_key_here" > .env
```

> `.env` は `.gitignore` に追加してください。API キーを git に含めないためです。

---

## よくあるトラブル

### `No space left on device`

共有メモリ不足。`docker-compose.yml` の `shm_size` が効いていない場合は、`--build` し直すか、`data.num_workers=0` で回避します。

```bash
PYTHONPATH=src python src/train.py data.num_workers=0
```

### GPU が認識されない

[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) がホストにインストールされているか確認してください。

```bash
# コンテナ内で確認
python -c "import torch; print(torch.cuda.is_available())"
```
