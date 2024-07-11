import logging
import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange

from netscripts.get_model import get_model
from netscripts.get_optimizer import get_optimizer

logger = logging.getLogger(__name__)


class VideoMAETrainer(pl.LightningModule):
    def __init__(self, cfg):
        super(VideoMAETrainer, self).__init__()
        self.cfg = cfg
        self.model = get_model(cfg)
        self.normalize_target = cfg.trainer.normalize_target
        self.patch_size = cfg.data_module.modality.patch_size[0]
        self.training_step_outputs = []

    def configure_optimizers(self):
        total_batch_size = self.scale_lr()
        self.trainer.fit_loop.setup_data()
        dataset = self.trainer.train_dataloader.dataset
        self.niter_per_epoch = len(dataset) // total_batch_size
        print("Number of training steps = %d" % self.niter_per_epoch)
        print(
            "Number of training examples per epoch = %d"
            % (total_batch_size * self.niter_per_epoch)
        )
        optimizer, scheduler = get_optimizer(
            self.cfg.trainer, self.model, self.niter_per_epoch
        )
        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, metric):
        cur_iter = self.trainer.global_step
        next_lr = scheduler.get_epoch_values(cur_iter + 1)[0]
        for param_group in self.trainer.optimizers[0].param_groups:
            param_group["lr"] = next_lr

    def normalize_videos(self, unnorm_videos):
        if self.normalize_target:
            videos_squeeze = rearrange(
                unnorm_videos,
                "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c",
                p0=2,
                p1=self.patch_size,
                p2=self.patch_size,
            )
            videos_norm = (
                videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
            ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
            # we find that the mean is about 0.48 and standard deviation is about 0.08.
            videos_patch = rearrange(videos_norm, "b n p c -> b n (p c)")
        else:
            videos_patch = rearrange(
                unnorm_videos,
                "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)",
                p0=2,
                p1=self.patch_size,
                p2=self.patch_size,
            )
        return videos_patch

    def training_step(self, batch, batch_idx):
        input = batch
        source_frames = input["source_frames"]
        unlabel_frames = input["unlabel_frames"]
        action_label = input["action_label"]
        bool_masked_pos = input["mask"]
        bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)

        with torch.no_grad():
            # calculate the predict label
            if self.cfg.data_module.modality.mode == "RGB":
                mean = torch.as_tensor(self.cfg.data_module.modality.mean)[
                    None, :, None, None, None
                ].type_as(source_frames)
                std = torch.as_tensor(self.cfg.data_module.modality.std)[
                    None, :, None, None, None
                ].type_as(source_frames)
            elif self.cfg.data_module.modality.mode == "flow":
                mean = torch.as_tensor(self.cfg.data_module.modality.mean)[
                    None, :, None, None, None
                ].type_as(source_frames)
                std = torch.as_tensor(self.cfg.data_module.modality.std)[
                    None, :, None, None, None
                ].type_as(source_frames)
            elif self.cfg.data_module.modality.mode == "pose":
                mean = torch.as_tensor(self.cfg.data_module.modality.mean)[
                    None, :, None, None, None
                ].type_as(source_frames)
                std = torch.as_tensor(self.cfg.data_module.modality.std)[
                    None, :, None, None, None
                ].type_as(source_frames)
            unnorm_videos_source = source_frames * std + mean  # in [0, 1]
            unnorm_videos_target = unlabel_frames * std + mean  # in [0, 1]

            videos_patch_source = self.normalize_videos(unnorm_videos_source)
            videos_patch_target = self.normalize_videos(unnorm_videos_target)

            B, _, C = videos_patch_source.shape
            labels_source = videos_patch_source[bool_masked_pos].reshape(B, -1, C)
            labels_target = videos_patch_target[bool_masked_pos].reshape(B, -1, C)

        preds_source, logits_source = self.model(source_frames, bool_masked_pos)
        preds_target, _ = self.model(unlabel_frames, bool_masked_pos)

        loss_mse = nn.MSELoss()
        loss_ce = nn.CrossEntropyLoss()
        recon_loss_source = loss_mse(input=preds_source, target=labels_source)
        recon_loss_target = loss_mse(input=preds_target, target=labels_target)
        ce_loss = loss_ce(logits_source, action_label)

        loss = recon_loss_source + self.cfg.trainer.modality.lambda_ce * ce_loss

        output = {
            "loss": loss.item(),
            "recon_loss_source": recon_loss_source.item(),
            "recon_loss_target": recon_loss_target.item(),
            "ce_loss": ce_loss.item(),
        }

        self.training_step_outputs.append(output)
        return loss

    def on_train_epoch_start(self):
        # shuffle the unlabel data loader
        unlabel_dir_to_img_frame = (
            self.trainer.train_dataloader.dataset.unlabel_loader._dir_to_img_frame
        )
        unlabel_start_frame = (
            self.trainer.train_dataloader.dataset.unlabel_loader._start_frame
        )
        lists = list(zip(unlabel_dir_to_img_frame, unlabel_start_frame))
        random.shuffle(lists)
        unlabel_dir_to_img_frame, unlabel_start_frame = zip(*lists)
        self.trainer.train_dataloader.dataset.unlabel_loader._dir_to_img_frame = list(
            unlabel_dir_to_img_frame
        )
        self.trainer.train_dataloader.dataset.unlabel_loader._start_frame = list(
            unlabel_start_frame
        )

    def on_train_epoch_end(self):
        train_loss = np.mean([output["loss"] for output in self.training_step_outputs])
        train_recon_loss_source = np.mean(
            [output["recon_loss_source"] for output in self.training_step_outputs]
        )
        train_recon_loss_target = np.mean(
            [output["recon_loss_target"] for output in self.training_step_outputs]
        )
        train_ce_loss = np.mean(
            [output["ce_loss"] for output in self.training_step_outputs]
        )
        self.log("train_loss", train_loss, on_step=False)
        self.log("train_recon_loss_source", train_recon_loss_source, on_step=False)
        self.log("train_recon_loss_target", train_recon_loss_target, on_step=False)
        self.log("train_ce_loss", train_ce_loss, on_step=False)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        self.training_step_outputs.clear()
        # save the model parameters
        if (self.trainer.current_epoch + 1) % self.cfg.save_ckpt_freq == 0:
            self.trainer.save_checkpoint(
                f"checkpoints/epoch={self.trainer.current_epoch:02d}-loss={train_loss:.4f}"
            )

    def validation_step(self, batch, batch_idx):
        return NotImplementedError

    def on_validation_epoch_end(self):
        return NotImplementedError

    def test_step(self, batch, batch_idx):
        return NotImplementedError

    def on_test_epoch_end(self):
        return NotImplementedError

    def scale_lr(self):
        total_batch_size = self.cfg.batch_size * len(self.cfg.devices)
        self.cfg.trainer.lr = self.cfg.trainer.lr * total_batch_size / 256
        self.cfg.trainer.min_lr = self.cfg.trainer.min_lr * total_batch_size / 256
        self.cfg.trainer.warmup_lr = self.cfg.trainer.warmup_lr * total_batch_size / 256
        print("LR = %.8f" % self.cfg.trainer.lr)
        print("Batch size = %d" % total_batch_size)
        return total_batch_size
