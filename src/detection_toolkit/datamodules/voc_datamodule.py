import torch
from torch.utils.data import DataLoader
import lightning as L
from torchvision import datasets, transforms
from typing import Optional, List, Tuple

class VOCDataModule(L.LightningDataModule):
    """
    Pascal VOC 2012 データセットを管理する DataModule。
    データのダウンロードから DataLoader の作成までをカプセル化します。
    """
    def __init__(self, data_dir: str = "./data", batch_size: int = 4, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # VOC 20 クラス（背景は含まない。class_id は 1〜20）
        self.classes = [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
            "train", "tvmonitor"
        ]
        self.class_to_idx = {cls: i + 1 for i, cls in enumerate(self.classes)}

    def prepare_data(self):
        """データのダウンロード（初回のみ実行）"""
        datasets.VOCDetection(self.data_dir, year="2012", image_set="train", download=True)
        datasets.VOCDetection(self.data_dir, year="2012", image_set="val", download=True)

    def setup(self, stage: Optional[str] = None):
        """学習・検証用データセットのインスタンス化"""
        self.train_ds = datasets.VOCDetection(
            self.data_dir, year="2012", image_set="train", transform=self.get_transforms()
        )
        self.val_ds = datasets.VOCDetection(
            self.data_dir, year="2012", image_set="val", transform=self.get_transforms()
        )

    def get_transforms(self):
        return transforms.Compose([
            transforms.ToTensor(),
        ])

    def collate_fn(self, batch):
        """画像ごとにボックス数が異なるため、Tensor ではなくリストでまとめる。"""
        images = []
        targets = []
        
        for img, label in batch:
            images.append(img)
            
            objs = label['annotation']['object']
            if not isinstance(objs, list):
                objs = [objs]
                
            boxes = []
            labels = []
            for obj in objs:
                bndbox = obj['bndbox']
                boxes.append([
                    float(bndbox['xmin']), float(bndbox['ymin']),
                    float(bndbox['xmax']), float(bndbox['ymax'])
                ])
                labels.append(self.class_to_idx[obj['name']])
            
            targets.append({
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64)
            })
            
        return images, targets

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=self.collate_fn
        )
