import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from netscripts.get_unlabel_loader import get_unlabel_loader

logger = logging.getLogger(__name__)


class Ego4DUnlabelCombinedMMDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transform, mask_gen):
        super(Ego4DUnlabelCombinedMMDataset, self).__init__()
        self.cfg = cfg
        self.transform_rgb = transform[0]
        self.transform_flow = transform[1]
        self.transform_pose = transform[2]
        self.mask_gen = mask_gen
        self._construct_source_loader(cfg)
        self._construct_unlabel_loader(cfg)

    def _construct_source_loader(self, cfg):
        # initialization
        self._clip_uid = []
        self._dir_to_img_frame = []
        self._clip_frame = []
        self._action_label = []
        self._action_list = []
        self._verb_list = []
        self._noun_list = []
        self._action_label_internal = []
        self._verb_label_internal = []
        self._noun_label_internal = []

        # read annotation json file
        with open(cfg.source_json_path) as f:
            data = json.load(f)
        for i, clip_dict in enumerate(data["clips"]):
            video_uid = clip_dict["video_id"]
            clip_uid = clip_dict["clip_uid"]
            clip_frame = clip_dict["clip_frame"]
            verb_label = clip_dict["verb_label"]
            noun_label = clip_dict["noun_label"]
            action_label = (verb_label, noun_label)

            # skip
            if video_uid in cfg.delete:
                print(
                    f"{video_uid} is invalid video, so it will not be included in the dataloarder"
                )
                continue

            if action_label not in self._action_list:
                self._action_list.append(action_label)
            if verb_label not in self._verb_list:
                self._verb_list.append(verb_label)
            if noun_label not in self._noun_list:
                self._noun_list.append(noun_label)

            action_label_internal = self._action_list.index(action_label)
            verb_label_internal = self._verb_list.index(verb_label)
            noun_label_internal = self._noun_list.index(noun_label)

            dir_to_img_frame = Path(cfg.source_data_dir, "image_frame", clip_uid)
            self._clip_uid.append(clip_uid)
            self._dir_to_img_frame.append(dir_to_img_frame)
            self._clip_frame.append(clip_frame)
            self._action_label.append(action_label)
            self._action_label_internal.append(action_label_internal)
            self._verb_label_internal.append(verb_label_internal)
            self._noun_label_internal.append(noun_label_internal)

        logger.info(f"Constructing Ego4D dataloader (size: {len(self._clip_frame)})")
        logger.info(f"Number of action classes: {len(self._action_list)}")

    def _construct_unlabel_loader(self, cfg):
        self.unlabel_loader = get_unlabel_loader(cfg.dataset)

    def _get_frame_source(self, dir_to_img_frame, frame_name, mode, frames):
        if mode == "RGB":
            path = dir_to_img_frame / Path(str(frame_name).zfill(6) + ".jpg")
            if path.exists():
                frame = Image.open(str(path))
            else:
                frame = frames[-1]
        elif mode == "flow":
            dir_to_flow_frame = str(dir_to_img_frame).replace(
                "image_frame", "optical_flow"
            )
            path = Path(dir_to_flow_frame, "npy", f"{str(frame_name).zfill(6)}.npy")
            if path.exists():
                frame = np.load(str(path))
            else:
                frame = frames[-1]
        elif mode == "pose":
            dir_to_pose_frame = str(dir_to_img_frame).replace(
                "image_frame", "hand-pose/heatmap"
            )
            path = Path(dir_to_pose_frame, f"{str(frame_name).zfill(6)}.npy")
            if path.exists():
                frame = np.load(str(path))
            else:
                frame = frames[-1]
        return frame

    def _get_frame_unlabel(self, dir_to_img_frame, frame_name, mode, frames):
        if mode == "RGB":
            path = dir_to_img_frame / Path(
                self.unlabel_loader.get_frame_str(frame_name)
            )
            if path.exists():
                frame = Image.open(str(path))
            else:
                frame = frames[-1]
        elif mode == "flow":
            dir_to_flow_frame = str(dir_to_img_frame).replace("rgb", "flow")
            path = Path(
                dir_to_flow_frame,
                self.unlabel_loader.get_frame_str(frame_name).replace("jpg", "npy"),
            )
            if path.exists():
                frame = np.load(str(path))
            else:
                frame = frames[-1]
        elif mode == "pose":
            dir_to_pose_frame = str(dir_to_img_frame).replace("rgb", "pose")
            path = Path(
                dir_to_pose_frame,
                self.unlabel_loader.get_frame_str(frame_name).replace("jpg", "npy"),
            )
            if path.exists():
                frame = np.load(str(path))
            else:
                frame = frames[-1]
        return frame

    def _get_input(
        self,
        source_dir_to_img_frame,
        source_clip_start_frame,
        unlabel_dir_to_img_frame,
        unlabel_clip_start_frame,
    ):
        # initialization
        # source_frames_rgb = []
        unlabel_frames_rgb = []
        unlabel_frames_flow = []
        unlabel_frames_pose = []

        # source_frame_names = [
        #     max(1, source_clip_start_frame + self.cfg.source_sampling_rate * i)
        #     for i in range(self.cfg.num_frames)
        # ]
        unlabel_frame_names = [
            max(1, unlabel_clip_start_frame + self.cfg.dataset.target_sampling_rate * i)
            for i in range(self.cfg.num_frames)
        ]

        # for frame_name in source_frame_names:
        #     source_frame_rgb = self._get_frame_source(
        #         source_dir_to_img_frame, frame_name, "RGB", source_frames_rgb
        #     )
        #     source_frames_rgb.append(source_frame_rgb)

        for frame_name in unlabel_frame_names:
            unlabel_frame_rgb = self._get_frame_unlabel(
                unlabel_dir_to_img_frame, frame_name, "RGB", unlabel_frames_rgb
            )
            unlabel_frames_rgb.append(unlabel_frame_rgb)
            unlabel_frame_flow = self._get_frame_unlabel(
                unlabel_dir_to_img_frame, frame_name, "flow", unlabel_frames_flow
            )
            unlabel_frames_flow.append(unlabel_frame_flow)
            unlabel_frame_pose = self._get_frame_unlabel(
                unlabel_dir_to_img_frame, frame_name, "pose", unlabel_frames_pose
            )
            unlabel_frames_pose.append(unlabel_frame_pose)

        # [T, H, W, C] -> [T*C, H, W] -> [C, T, H, W]
        # source_frames_rgb = self.transform_rgb.weak_aug(source_frames_rgb)
        # source_frames_rgb = source_frames_rgb.permute(1, 0, 2, 3)
        unlabel_frames_rgb = self.transform_rgb.weak_aug(unlabel_frames_rgb)
        unlabel_frames_rgb = unlabel_frames_rgb.permute(1, 0, 2, 3)

        unlabel_frames_flow = self.transform_flow.weak_aug(unlabel_frames_flow)
        unlabel_frames_flow = unlabel_frames_flow.permute(1, 0, 2, 3)

        unlabel_frames_pose = self.transform_pose.weak_aug(unlabel_frames_pose)
        unlabel_frames_pose = unlabel_frames_pose.permute(1, 0, 2, 3)

        # mask generation
        mask = self.mask_gen()

        return (
            # source_frames_rgb,
            unlabel_frames_rgb,
            unlabel_frames_flow,
            unlabel_frames_pose,
            mask,
        )

    def __getitem__(self, index):
        input = {}

        # source
        source_dir_to_img_frame = self._dir_to_img_frame[index]
        source_clip_start_frame = self._clip_frame[index]
        # unlabel
        unlabel_index = index % len(self.unlabel_loader)
        unlabel_dir_to_img_frame = self.unlabel_loader._dir_to_img_frame[unlabel_index]
        unlabel_clip_start_frame = self.unlabel_loader._start_frame[unlabel_index]

        (
            # source_frames_rgb,
            unlabel_frames_rgb,
            unlabel_frames_flow,
            unlabel_frames_pose,
            mask,
        ) = self._get_input(
            source_dir_to_img_frame,
            source_clip_start_frame,
            unlabel_dir_to_img_frame,
            unlabel_clip_start_frame,
        )

        # label
        action_label_internal = self._action_label_internal[index]

        # input["source_frames_rgb"] = source_frames_rgb
        input["unlabel_frames_rgb"] = unlabel_frames_rgb
        input["unlabel_frames_flow"] = unlabel_frames_flow
        input["unlabel_frames_pose"] = unlabel_frames_pose
        input["action_label"] = action_label_internal
        input["mask"] = mask
        return input

    def __len__(self):
        return len(self._clip_frame)
