import torch
import lightning as L
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import wandb

# Tensor Core を有効化して RTX シリーズでの行列演算を高速化
torch.set_float32_matmul_precision('medium')

class VOCDetector(L.LightningModule):
    """Faster R-CNN (MobileNetV3-Large-320-FPN) による物体検出モデル。"""
    def __init__(self, num_classes: int = 21, lr: float = 0.005, momentum: float = 0.9, weight_decay: float = 0.0005):
        super().__init__()
        self.save_hyperparameters()

        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        )

        # 出力層をクラス数に合わせて差し替え
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # mAP 評価指標（COCO 形式: IoU 0.50:0.95、VOC 形式: IoU 0.50）
        self.val_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)  # 学習時は loss 辞書を返す
        losses = sum(loss for loss in loss_dict.values())
        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True)
        self.log("train_loss", losses, on_step=True, on_epoch=True, prog_bar=True)
        
        return losses

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        # --- 損失計算 ---
        # Faster R-CNN の仕様: eval モードでは loss が返らないため train モードで実行
        self.model.train()
        with torch.no_grad():
            loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log("val_loss", losses, on_epoch=True, prog_bar=True)

        # --- 予測（mAP 計算 & 画像ログ用）---
        # eval モードに切り替えて予測を取得（BatchNorm が推論モードになる）
        self.model.eval()
        with torch.no_grad():
            preds = self.model(images)

        # mAP メトリクスの蓄積（エポック末の on_validation_epoch_end で compute する）
        self.val_map.update(preds, targets)

        # バッチ 0 の先頭 1 枚を wandb に画像ログ
        if batch_idx == 0 and self.logger is not None:
            img = images[0].cpu().permute(1, 2, 0).numpy()
            pred_boxes = preds[0]['boxes'].cpu().numpy()
            pred_labels = preds[0]['labels'].cpu().numpy()
            pred_scores = preds[0]['scores'].cpu().numpy()
            mask = pred_scores > 0.5
            wandb_img = wandb.Image(img, boxes={
                "predictions": {
                    "box_data": [
                        {"position": {"minX": float(b[0]), "minY": float(b[1]),
                                      "maxX": float(b[2]), "maxY": float(b[3])},
                         "class_id": int(l), "scores": {"score": float(s)}}
                        for b, l, s in zip(pred_boxes[mask], pred_labels[mask], pred_scores[mask])
                    ]
                }
            })
            self.logger.experiment.log({"val_predictions": wandb_img})

        return losses

    def on_validation_epoch_end(self):
        """エポック末に mAP を計算して wandb にログし、メトリクスをリセットする。"""
        map_metrics = self.val_map.compute()
        self.log("val_mAP",    map_metrics["map"],    prog_bar=True)
        self.log("val_mAP_50", map_metrics["map_50"], prog_bar=False)
        self.log("val_mAP_75", map_metrics["map_75"], prog_bar=False)
        self.val_map.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return [optimizer], [lr_scheduler]
