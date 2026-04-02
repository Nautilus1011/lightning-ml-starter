import torch
import lightning as L
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import wandb

# Tensor Core を有効化して RTX シリーズでの行列演算を高速化
torch.set_float32_matmul_precision('medium')

class VOCDetector(L.LightningModule):
    """
    Faster R-CNN (MobileNetV3-Large-320-FPN バックボーン) を使った物体検出モデル。
    ResNet50 FPN 版と比べて重みが約 1/10、学習速度が約 3〜4 倍速いため
    スペックが限られたマシンでも快適に実験できます。
    学習、バリデーション、ロギングのロジックを管理します。
    """
    def __init__(self, num_classes: int = 21, lr: float = 0.005, momentum: float = 0.9, weight_decay: float = 0.0005):
        super().__init__()
        self.save_hyperparameters()

        # Pre-trainedモデルのロード（COCO で事前学習済みの軽量バックボーンを使用）
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        )

        # 出力層のカスタマイズ（背景を含めたクラス数に合わせる）
        # ResNet50 版と完全に同じ API なので、この行は変更不要
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        # Faster R-CNNは学習時、imagesとtargetsを渡すとLossの辞書を返す
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # ログ出力。on_step=Trueでグラフが細かく、on_epoch=Trueで全体傾向が見える
        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True)
        self.log("train_loss", losses, on_step=True, on_epoch=True, prog_bar=True)
        
        return losses

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        
        # 検証時もLossを計算するために一時的にモデルをtrainモードにする
        # (Faster R-CNNの仕様上、evalモードではLossが計算されないため)
        self.model.train()
        with torch.no_grad():
            loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log("val_loss", losses, on_epoch=True, prog_bar=True)

        # --- 可視化ログ (研究加速機能) ---
        # 最初のバッチの数枚だけ推論結果を WandB に画像として送る
        if batch_idx == 0 and self.logger is not None:
            self.model.eval()
            with torch.no_grad():
                preds = self.model(images)
            
            # wandb.Image を作成（簡略化のため最初の1枚のみ）
            img = images[0].cpu().permute(1, 2, 0).numpy()
            pred_boxes = preds[0]['boxes'].cpu().numpy()
            pred_labels = preds[0]['labels'].cpu().numpy()
            pred_scores = preds[0]['scores'].cpu().numpy()

            # スコアが一定以上のものだけをログに残す
            mask = pred_scores > 0.5
            wandb_img = wandb.Image(img, boxes={
                "predictions": {
                    "box_data": [
                        {"position": {"minX": b[0], "minY": b[1], "maxX": b[2], "maxY": b[3]},
                         "class_id": int(l), "scores": {"score": float(s)}}
                        for b, l, s in zip(pred_boxes[mask], pred_labels[mask], pred_scores[mask])
                    ]
                }
            })
            self.logger.experiment.log({"val_predictions": wandb_img})
            self.model.train() # 元に戻す

        return losses

    def configure_optimizers(self):
        """
        Optimizer と学習率スケジューラの設定。
        大学の研究では、まずは SGD から始めるのが定番です。
        """
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay
        )
        # StepLR: 規定のエポックごとに学習率を下げる。これにより学習の停滞を防ぐ
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return [optimizer], [lr_scheduler]
