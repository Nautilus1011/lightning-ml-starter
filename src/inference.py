"""
inference.py — 学習済みモデルを使った物体検出推論スクリプト。

実行ごとに outputs/runs/YYYY-MM-DD/HH-MM-SS/ が自動生成されます。
そのディレクトリの中に結果画像と wandb ログがまとまります。

使い方:
  # 単一画像
  cd /app
  PYTHONPATH=src python src/inference.py \\
    --checkpoint outputs/train/YYYY-MM-DD/HH-MM-SS/checkpoints/best-checkpoint.ckpt \\
    --image data/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg

  # ディレクトリ内の全画像を一括処理
  PYTHONPATH=src python src/inference.py \\
    --checkpoint outputs/train/YYYY-MM-DD/HH-MM-SS/checkpoints/best-checkpoint.ckpt \\
    --image_dir data/VOCdevkit/VOC2012/JPEGImages \\
    --num_images 20
"""

import os
import glob
import argparse
from datetime import datetime
from pathlib import Path

import torch
import wandb
import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI 不要のバックエンド（サーバー・コンテナ環境向け）
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision.transforms as T

from detection_toolkit.models.detector import VOCDetector

# VOC 20クラスのラベル名（class_id 1〜20 に対応）
VOC_CLASSES = [
    "__background__",
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# クラスごとに色を固定して見やすくする
COLORS = plt.colormaps.get_cmap("tab20")(np.linspace(0, 1, len(VOC_CLASSES)))


def draw_predictions(img_pil: Image.Image, boxes, labels, scores, score_thresh: float = 0.5):
    """
    バウンディングボックスとラベルを画像に描画して返す。

    研究でよくある「推論結果を目視確認したい」というニーズに応えるための関数。
    score_thresh 以上のスコアを持つ予測だけを表示します。
    """
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_pil)
    ax.axis("off")

    for box, label, score in zip(boxes, labels, scores):
        if score < score_thresh:
            continue
        xmin, ymin, xmax, ymax = box
        cls_name = VOC_CLASSES[label] if label < len(VOC_CLASSES) else str(label)
        color = COLORS[label % len(COLORS)]

        # バウンディングボックスの描画
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

        # ラベルとスコアの表示
        ax.text(
            xmin, max(ymin - 5, 0),
            f"{cls_name}: {score:.2f}",
            color="white", fontsize=9, fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.7, pad=2, edgecolor="none"),
        )

    plt.tight_layout(pad=0)
    return fig


def run_inference_on_image(model, img_path: str, device, score_thresh: float, output_dir: Path):
    """
    1枚の画像に対して推論を行い、結果画像を保存して wandb.Image を返す。
    """
    img_pil = Image.open(img_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img_pil).to(device)

    with torch.no_grad():
        # Faster R-CNN はリストで渡す必要がある（重要: unsqueeze ではない）
        preds = model([img_tensor])

    boxes  = preds[0]["boxes"].cpu().numpy()
    labels = preds[0]["labels"].cpu().numpy()
    scores = preds[0]["scores"].cpu().numpy()

    n_detected = int((scores >= score_thresh).sum())

    # バウンディングボックス付き画像を保存
    fig = draw_predictions(img_pil, boxes, labels, scores, score_thresh)
    save_path = output_dir / (Path(img_path).stem + "_pred.jpg")
    fig.savefig(save_path, dpi=80, bbox_inches="tight")
    plt.close(fig)

    # wandb.Image 形式に変換（ダッシュボードで画像として表示される）
    wandb_boxes = [
        {
            "position": {
                "minX": float(b[0]), "minY": float(b[1]),
                "maxX": float(b[2]), "maxY": float(b[3]),
            },
            "class_id": int(l),
            "scores": {"score": float(s)},
        }
        for b, l, s in zip(boxes, labels, scores)
        if s >= score_thresh
    ]
    wandb_img = wandb.Image(
        str(save_path),
        caption=f"{Path(img_path).name} — {n_detected} objects",
        boxes={
            "predictions": {
                "box_data": wandb_boxes,
                "class_labels": {i: c for i, c in enumerate(VOC_CLASSES)},
            }
        },
    )

    return n_detected, wandb_img


def main():
    _now = datetime.now()
    default_output_dir = f"outputs/runs/{_now.strftime('%Y-%m-%d/%H-%M-%S')}"

    parser = argparse.ArgumentParser(description="VOC 物体検出 推論スクリプト")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="学習済みチェックポイントのパス (.ckpt)")
    # 単体 or ディレクトリ指定のどちらでも動く
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",      type=str, help="推論する画像のパス（単体）")
    group.add_argument("--image_dir",  type=str, help="画像が入ったディレクトリのパス（一括処理）")
    parser.add_argument("--num_images",  type=int, default=20,
                        help="--image_dir 指定時に処理する画像数 (default: 20)")
    parser.add_argument("--score_thresh", type=float, default=0.5,
                        help="表示する予測のスコア閾値 (default: 0.5)")
    parser.add_argument("--output_dir", type=str, default=default_output_dir,
                        help=f"結果の保存先 (default: {default_output_dir})")
    parser.add_argument("--wandb_project", type=str, default="voc-object-detection",
                        help="wandb プロジェクト名")
    args = parser.parse_args()

    # 出力ディレクトリの作成
    # outputs/runs/YYYY-MM-DD/HH-MM-SS/ に結果画像と wandb ログがまとまる
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"出力ディレクトリ: {output_dir.resolve()}")

    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # モデルのロード
    print(f"チェックポイントを読み込み中: {args.checkpoint}")
    model = VOCDetector.load_from_checkpoint(args.checkpoint)
    model.to(device)
    model.eval()

    # 推論する画像リストを作成
    if args.image:
        image_paths = [args.image]
    else:
        all_images = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")))
        image_paths = all_images[: args.num_images]
        print(f"{len(all_images)} 枚中 {len(image_paths)} 枚を処理します")

    # ─── wandb の推論専用 run を開始 ───────────────────────────────
    # job_type="inference" を設定することで、学習 run (job_type="train") と
    # ダッシュボード上でフィルタリングして分けて見ることができる。
    # dir=str(output_dir) でこの実行のディレクトリ内に wandb/ を作成
    run = wandb.init(
        project=args.wandb_project,
        job_type="inference",          # ← 学習 run との識別キー
        name=f"inference_{Path(args.checkpoint).stem}",
        dir=str(output_dir),
        config={
            "checkpoint": args.checkpoint,
            "score_thresh": args.score_thresh,
            "num_images": len(image_paths),
        },
        save_code=False,
    )

    # 推論ループ
    results_table = wandb.Table(columns=["image_name", "n_detected", "prediction"])
    total_detected = 0

    for i, img_path in enumerate(image_paths):
        n_det, wandb_img = run_inference_on_image(
            model, img_path, device, args.score_thresh, output_dir
        )
        total_detected += n_det
        results_table.add_data(Path(img_path).name, n_det, wandb_img)
        print(f"[{i+1:>3}/{len(image_paths)}] {Path(img_path).name} — {n_det} objects detected")

    # wandb にまとめてログ
    run.log({
        "inference_results": results_table,
        "total_images": len(image_paths),
        "total_detections": total_detected,
        "avg_detections_per_image": total_detected / max(len(image_paths), 1),
    })
    run.finish()

    print(f"\n完了: {len(image_paths)} 枚処理, 合計 {total_detected} 個の物体を検出")
    print(f"結果画像の保存先: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

