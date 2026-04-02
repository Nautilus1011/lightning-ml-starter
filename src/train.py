import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

# パッケージ化されているため、絶対パスでインポート可能です
from detection_toolkit.datamodules.voc_datamodule import VOCDataModule
from detection_toolkit.models.detector import VOCDetector

@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    """
    学習の実行メイン関数。
    """
    print(OmegaConf.to_yaml(cfg))

    # WandB ロガー
    wandb_logger = WandbLogger(
        project=cfg.logger.project_name,
        name=cfg.logger.experiment_name,
        save_dir="outputs/"
    )

    # DataModule
    dm = VOCDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers
    )

    # Model
    model = VOCDetector(
        num_classes=cfg.model.num_classes,
        lr=cfg.model.lr,
        momentum=cfg.model.momentum,
        weight_decay=cfg.model.weight_decay
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="outputs/checkpoints",
        monitor=cfg.callbacks.model_checkpoint.monitor,
        mode=cfg.callbacks.model_checkpoint.mode,
        save_top_k=cfg.callbacks.model_checkpoint.save_top_k,
        filename=cfg.callbacks.model_checkpoint.filename,
    )
    early_stop_callback = EarlyStopping(
        monitor=cfg.callbacks.early_stopping.monitor,
        patience=cfg.callbacks.early_stopping.patience,
        mode=cfg.callbacks.early_stopping.mode,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Trainer
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        log_every_n_steps=cfg.trainer.log_every_n_steps
    )

    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    train()
