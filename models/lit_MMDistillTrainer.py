import copy
import logging
import random

import numpy as np
import pytorch_lightning as pl
import sklearn.linear_model
import torch
import torch.nn as nn
import torchmetrics
from einops import rearrange
from torch.nn.functional import normalize

from models.cmt import CrossModalTranslate
from netscripts.get_model import get_model
from netscripts.get_optimizer import get_optimizer_mmdistill

logger = logging.getLogger(__name__)


class MMDistillTrainer(pl.LightningModule):
    def __init__(self, cfg):
        super(MMDistillTrainer, self).__init__()
        self.cfg = cfg
        # model
        self.student_rgb = get_model(
            cfg,
            ckpt_pth=cfg.trainer.ckpt_path[0],
            input_size=cfg.data_module.input_size[0],
            patch_size=cfg.data_module.patch_size[0][0],
            in_chans=cfg.trainer.in_chans[0],
        )
        self.teacher_flow = get_model(
            cfg,
            ckpt_pth=cfg.trainer.ckpt_path[1],
            input_size=cfg.data_module.input_size[1],
            patch_size=cfg.data_module.patch_size[1][0],
            in_chans=cfg.trainer.in_chans[1],
        )
        self.teacher_pose = get_model(
            cfg,
            ckpt_pth=cfg.trainer.ckpt_path[2],
            input_size=cfg.data_module.input_size[2],
            patch_size=cfg.data_module.patch_size[2][0],
            in_chans=cfg.trainer.in_chans[2],
        )
        self.cmt = CrossModalTranslate()
        self.teacher_rgb = copy.deepcopy(self.student_rgb)
        self.teacher_rgb.requires_grad_(False)
        self.teacher_flow.requires_grad_(False)
        self.teacher_pose.requires_grad_(False)

        # loss
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.train_top1_a = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.cfg.data_module.num_classes_action
        )

        # initialization
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def configure_optimizers(self):
        self.scale_lr()
        self.trainer.fit_loop.setup_data()
        dataset = self.trainer.train_dataloader.dataset
        self.niter_per_epoch = len(dataset) // self.total_batch_size
        print("Number of training steps = %d" % self.niter_per_epoch)
        print(
            "Number of training examples per epoch = %d"
            % (self.total_batch_size * self.niter_per_epoch)
        )
        optimizer, scheduler = get_optimizer_mmdistill(
            self.cfg.trainer, [self.student_rgb, self.cmt], self.niter_per_epoch
        )
        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, metric):
        cur_iter = self.trainer.global_step
        next_lr = scheduler.get_epoch_values(cur_iter + 1)[0]
        for param_group in self.trainer.optimizers[0].param_groups:
            param_group["lr"] = next_lr

    def _forward_loss_action(
        self,
        unlabel_frames_rgb_w,
        unlabel_frames_flow_w,
        unlabel_frames_pose_w,
        mask=None,
    ):
        # feature distillation
        fr, _ = self.teacher_rgb(unlabel_frames_rgb_w, mask)
        ff, _ = self.teacher_flow(unlabel_frames_flow_w, mask)
        fp, _ = self.teacher_pose(unlabel_frames_pose_w, mask)
        x_rgb, _ = self.student_rgb(unlabel_frames_rgb_w, mask)
        trans_rgb, trans_flow, trans_pose = self.cmt(x_rgb)

        trans_loss_rgb = self.mse_loss(trans_rgb, fr.detach())
        trans_loss_flow = self.mse_loss(trans_flow, ff.detach())
        trans_loss_pose = self.mse_loss(trans_pose, fp.detach())
        return trans_loss_rgb, trans_loss_flow, trans_loss_pose

    def training_step(self, batch, batch_idx):
        input = batch

        unlabel_frames_rgb_w = input["unlabel_frames_rgb"]
        unlabel_frames_flow_w = input["unlabel_frames_flow"]
        unlabel_frames_pose_w = input["unlabel_frames_pose"]
        bool_masked_pos = input["mask"]
        bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)

        trans_loss_rgb, trans_loss_flow, trans_loss_pose = self._forward_loss_action(
            unlabel_frames_rgb_w,
            unlabel_frames_flow_w,
            unlabel_frames_pose_w,
            bool_masked_pos,
        )

        loss = trans_loss_rgb + trans_loss_flow + trans_loss_pose

        outputs = {
            "train_loss": loss.item(),
            "trans_loss_rgb": trans_loss_rgb.item(),
            "trans_loss_flow": trans_loss_flow.item(),
            "trans_loss_pose": trans_loss_pose.item(),
        }

        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        self.log_dict(outputs)
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

    def validation_step(self, batch, batch_idx):
        input = batch[0]

        frames_rgb = input["frames"]
        action_idx = input["action_idx"]
        bool_masked_pos = input["mask"]
        bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)

        # convert labels for fewshot evaluation
        _, action_idx = torch.unique(action_idx, return_inverse=True)

        n_way = self.cfg.data_module.n_way
        k_shot = self.cfg.data_module.k_shot
        q_sample = self.cfg.data_module.q_sample

        # RGB
        frames_rgb, support_frames_rgb, query_frames_rgb = self.preprocess_frames(
            frames=frames_rgb, n_way=n_way, k_shot=k_shot, q_sample=q_sample
        )

        # mask
        support_mask = bool_masked_pos[: k_shot * n_way]
        query_mask = bool_masked_pos[k_shot * n_way :]

        action_idx = action_idx.view(n_way, (k_shot + q_sample))
        support_action_label, query_action_label = (
            action_idx[:, :k_shot].flatten(),
            action_idx[:, k_shot:].flatten(),
        )

        pred_rgb, prob_rgb = self.LR(
            self.student_rgb,
            support=support_frames_rgb,
            support_label=support_action_label,
            query=query_frames_rgb,
            support_mask=support_mask,
            query_mask=query_mask,
        )

        acc = torchmetrics.Accuracy(task="multiclass", num_classes=5)

        top1_action = acc(pred_rgb.cpu(), query_action_label.cpu())

        outputs = {
            "top1_action": top1_action.item(),
        }
        self.validation_step_outputs.append(outputs)

    def on_validation_epoch_end(self):
        top1_action = np.mean(
            [output["top1_action"] for output in self.validation_step_outputs]
        )
        self.log("val_top1_action", top1_action, on_step=False)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        input = batch[0]

        frames_rgb = input["frames"]
        action_idx = input["action_idx"]
        bool_masked_pos = input["mask"]
        bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)

        # convert labels for fewshot evaluation
        _, action_idx = torch.unique(action_idx, return_inverse=True)

        n_way = self.cfg.data_module.n_way
        k_shot = self.cfg.data_module.k_shot
        q_sample = self.cfg.data_module.q_sample

        # RGB
        frames_rgb, support_frames_rgb, query_frames_rgb = self.preprocess_frames(
            frames=frames_rgb, n_way=n_way, k_shot=k_shot, q_sample=q_sample
        )

        # mask
        support_mask = bool_masked_pos[: k_shot * n_way]

        query_masks = []
        for _ in range(2):
            query_mask = bool_masked_pos[k_shot * n_way :]
            query_masks.append(query_mask)
            # Shift by 1 in the batch dimension
            bool_masked_pos = torch.cat(
                (bool_masked_pos[1:], bool_masked_pos[:1]), dim=0
            )

        action_idx = action_idx.view(n_way, (k_shot + q_sample))
        support_action_label, query_action_label = (
            action_idx[:, :k_shot].flatten(),
            action_idx[:, k_shot:].flatten(),
        )

        # # prediction with no mask
        # pred_rgb, prob_rgb = self.LR(
        #     self.student_rgb,
        #     support=support_frames_rgb,
        #     support_label=support_action_label,
        #     query=query_frames_rgb,
        # )

        # prediction with mask and ensemble
        pred_rgb_ensemble, prob_rgb_original = self.LR_ensemble(
            self.teacher_rgb,
            support=support_frames_rgb,
            support_label=support_action_label,
            query=query_frames_rgb,
            support_mask=support_mask,
            query_masks=query_masks[:2],
        )

        acc = torchmetrics.Accuracy(task="multiclass", num_classes=5)

        # top1_action = acc(pred_rgb.cpu(), query_action_label.cpu())
        top1_action_ensemble = acc(pred_rgb_ensemble.cpu(), query_action_label.cpu())

        outputs = {
            # "top1_action": top1_action.item(),
            "top1_action_ensemble": top1_action_ensemble.item(),
        }
        self.test_step_outputs.append(outputs)

    def on_test_epoch_end(self):
        top1_action_ensemble = np.mean(
            [output["top1_action_ensemble"] for output in self.test_step_outputs]
        )
        top1_action_ensemble_std = np.std(
            [output["top1_action_ensemble"] for output in self.test_step_outputs]
        )
        top1_action_ensemble_std_error = top1_action_ensemble_std / np.sqrt(
            len(self.test_step_outputs)
        )
        self.log("top1_action_ensemble", top1_action_ensemble, on_step=False)
        self.log("top1_action_ensemble_std", top1_action_ensemble_std, on_step=False)
        self.log(
            "top1_action_ensemble_std_error",
            top1_action_ensemble_std_error,
            on_step=False,
        )

    def scale_lr(self):
        self.total_batch_size = self.cfg.batch_size * len(self.cfg.devices)
        self.cfg.trainer.lr = self.cfg.trainer.lr * self.total_batch_size / 256
        self.cfg.trainer.min_lr = self.cfg.trainer.min_lr * self.total_batch_size / 256
        self.cfg.trainer.warmup_lr = (
            self.cfg.trainer.warmup_lr * self.total_batch_size / 256
        )
        print("LR = %.8f" % self.cfg.trainer.lr)
        print("Batch size = %d" % self.total_batch_size)

    def preprocess_frames(self, frames, n_way, k_shot, q_sample):
        frames = rearrange(
            frames, "(n m) c t h w -> n m c t h w", n=n_way, m=(k_shot + q_sample)
        )

        support_frames = rearrange(
            frames[:, :k_shot],
            "n m c t h w -> (n m) c t h w",
            n=n_way,
            m=k_shot,
        )
        query_frames = rearrange(
            frames[:, k_shot:],
            "n m c t h w -> (n m) c t h w",
            n=n_way,
            m=q_sample,
        )
        return frames, support_frames, query_frames

    @torch.no_grad()
    def LR(
        self,
        model,
        support,
        support_label,
        query,
        support_mask=None,
        query_mask=None,
        norm=False,
    ):
        """logistic regression classifier"""
        support = model(support, support_mask)[0].detach()
        query = model(query, query_mask)[0].detach()
        if norm:
            support = normalize(support)
            query = normalize(query)

        clf = sklearn.linear_model.LogisticRegression(
            random_state=0,
            solver="lbfgs",
            max_iter=1000,
            C=1,
            multi_class="multinomial",
        )

        support_features_np = support.data.cpu().numpy()
        support_label_np = support_label.data.cpu().numpy()
        clf.fit(support_features_np, support_label_np)

        query_features_np = query.data.cpu().numpy()
        pred = clf.predict(query_features_np)
        prob = clf.predict_proba(query_features_np)

        pred = torch.from_numpy(pred).type_as(support)
        prob = torch.from_numpy(prob).type_as(support)
        return pred, prob

    @torch.no_grad()
    def LR_ensemble(
        self,
        model,
        support,
        support_label,
        query,
        support_mask=None,
        query_masks=None,
        norm=False,
    ):
        """logistic regression classifier"""
        support = model(support, support_mask)[0].detach()

        clf = sklearn.linear_model.LogisticRegression(
            random_state=0,
            solver="lbfgs",
            max_iter=1000,
            C=1,
            multi_class="multinomial",
        )

        support_features_np = support.data.cpu().numpy()
        support_label_np = support_label.data.cpu().numpy()
        clf.fit(support_features_np, support_label_np)

        probs = []
        for query_mask in query_masks:
            query_features = model(query, query_mask)[0].detach()

            query_features_np = query_features.data.cpu().numpy()
            prob = clf.predict_proba(query_features_np)
            probs.append(prob)

        probs = np.array(probs)
        prob = np.mean(probs, axis=0)
        pred = np.argmax(prob, axis=1)
        pred = torch.from_numpy(pred).type_as(support)
        prob = torch.from_numpy(prob).type_as(support)
        return pred, prob
