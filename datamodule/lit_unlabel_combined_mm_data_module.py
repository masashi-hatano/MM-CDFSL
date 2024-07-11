import pytorch_lightning as pl
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import BatchSampler

from datamodule.dataset.ego4d_unlabel_combined_multimodal_dataset import (
    Ego4DUnlabelCombinedMMDataset,
)
from datamodule.utils.augmentation import (
    DataAugmentationForUnlabelMM,
    DataAugmentationForUnlabelRGB,
    DataAugmentationForVideoMAERGB,
    MaskGeneration,
)
from datamodule.utils.episodic_batch_sampler import EpisodicBatchSampler
from netscripts.get_fewshot_eval_dataset import get_fewshot_eval_dataset


class UnlabelCombinedMMDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super(UnlabelCombinedMMDataModule, self).__init__()
        self.cfg = cfg
        self.data_module_cfg = cfg.data_module
        self.n_way = cfg.data_module.n_way
        self.k_shot = cfg.data_module.k_shot
        self.q_sample = cfg.data_module.q_sample
        self.episodes = cfg.data_module.episodes
        self.eval_batch_size = self.n_way * (self.k_shot + self.q_sample)
        # transform
        self.mask_gen = MaskGeneration(cfg.data_module)
        self.transform_train_rgb = DataAugmentationForUnlabelRGB(cfg)
        self.transform_train_flow = DataAugmentationForUnlabelMM(
            cfg, mean=cfg.data_module.mean[1], std=cfg.data_module.std[1]
        )
        self.transform_train_pose = DataAugmentationForUnlabelMM(
            cfg.data_module, mean=cfg.data_module.mean[2], std=cfg.data_module.std[2]
        )
        self.transform_eval_rgb = DataAugmentationForVideoMAERGB(
            cfg.data_module, multi_scale_crop=False
        )
        self.transform_train = [
            self.transform_train_rgb,
            self.transform_train_flow,
            self.transform_train_pose,
        ]
        self.transform_eval = self.transform_eval_rgb

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = Ego4DUnlabelCombinedMMDataset(
                self.data_module_cfg, self.transform_train, self.mask_gen
            )
            self.batch_sampler_train = BatchSampler(
                sampler=DistributedSampler(
                    dataset=self.train_dataset,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=True,
                ),
                batch_size=self.cfg.batch_size,
                drop_last=True,
            )
            self.val_dataset = get_fewshot_eval_dataset(
                self.data_module_cfg.dataset,
                self.transform_eval,
                self.mask_gen,
                self.data_module_cfg.num_frames,
                "RGB",
            )
            batch_sampler = BatchSampler(
                sampler=DistributedSampler(
                    dataset=range(self.eval_batch_size * self.episodes),
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                ),
                batch_size=self.eval_batch_size,
                drop_last=False,
            )
            self.episodic_batch_sampler_val = EpisodicBatchSampler(
                dataset=self.val_dataset,
                batch_sampler=batch_sampler,
                n_way=self.n_way,
                k_shot=self.k_shot,
                q_sample=self.q_sample,
                episodes=self.episodes,
            )
        elif stage == "test":
            self.test_dataset = get_fewshot_eval_dataset(
                self.data_module_cfg.dataset,
                self.transform_eval,
                self.mask_gen,
                self.data_module_cfg.num_frames,
                "RGB",
            )
            self.episodic_batch_sampler_test = EpisodicBatchSampler(
                dataset=self.test_dataset,
                n_way=self.n_way,
                k_shot=self.k_shot,
                q_sample=self.q_sample,
                episodes=self.episodes,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.batch_sampler_train,
            num_workers=self.cfg.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_sampler=self.episodic_batch_sampler_val,
            num_workers=self.cfg.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_sampler=self.episodic_batch_sampler_test,
            num_workers=self.cfg.num_workers,
        )
