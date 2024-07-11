import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datamodule.dataset.ego4d_unlabel_combined_dataset import (
    Ego4DUnlabelCombinedDataset,
)
from datamodule.utils.augmentation import (
    DataAugmentationForUnlabelMM,
    DataAugmentationForUnlabelRGB,
    MaskGeneration,
)


class UnlabelCombinedPretrainDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super(UnlabelCombinedPretrainDataModule, self).__init__()
        self.cfg = cfg
        self.data_module_cfg = cfg.data_module
        self.mode = cfg.data_module.modality.mode
        self.mask_gen = MaskGeneration(cfg.data_module)
        if self.mode == "RGB":
            self.transform_train = DataAugmentationForUnlabelRGB(
                cfg, input_size=cfg.data_module.modality.input_size
            )
        elif self.mode == "flow" or self.mode == "pose":
            self.transform_train = DataAugmentationForUnlabelMM(
                cfg,
                mean=cfg.data_module.modality.mean,
                std=cfg.data_module.modality.std,
            )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = Ego4DUnlabelCombinedDataset(
                self.data_module_cfg,
                self.transform_train,
                self.mask_gen,
                mode=self.mode,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            drop_last=True,
        )
