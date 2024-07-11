import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class WEARFewshotEvalDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transform, mask_gen, num_frames, mode):
        super(WEARFewshotEvalDataset, self).__init__()
        self.cfg = cfg
        self.transform = transform
        self.mask_gen = mask_gen
        self.mode = mode
        self._construct_loader(cfg)

    def _construct_loader(self, cfg):
        # initialization
        self._dir_to_img_frame = []
        self._video_id = []
        self._start_frame = []
        self._action_label = []
        self._action_idx = []

        # read annotation json file
        with open(cfg.fewshot_eval_json_path) as f:
            data = json.load(f)
        assert data["split"] == "val"

        self.num_actions = data["num_actions"]

        for i, clip_dict in enumerate(data["clips"]):
            video_id = clip_dict["video_id"]
            start_frame = int(clip_dict["start_frame"])
            action_idx = clip_dict["label_id_internal"]
            action_label = clip_dict["label"]

            dir_to_img_frame = Path(
                cfg.target_data_dir,
                "RGB_frames",
                video_id,
            )
            self._dir_to_img_frame.append(dir_to_img_frame)
            self._start_frame.append(start_frame)
            self._action_label.append(action_label)
            self._action_idx.append(action_idx)

        logger.info(f"Constructing WEAR dataloader (size: {len(self._start_frame)})")

    def _get_frame(self, dir_to_img_frame, frame_name, mode, frames):
        if mode == "RGB":
            path = dir_to_img_frame / Path(f"{str(frame_name).zfill(6)}.jpg")
            if path.exists():
                frame = Image.open(str(path))
            else:
                frame = frames[-1]
        elif mode == "flow":
            dir_to_flow_frame = str(dir_to_img_frame).replace("RGB", "flow")
            path = Path(dir_to_flow_frame, f"{str(frame_name).zfill(6)}.npy")
            if path.exists():
                frame = np.load(str(path))
            else:
                frame = frames[-1]
        elif mode == "pose":
            dir_to_flow_frame = str(dir_to_img_frame).replace(
                "RGB_frames", "hand-pose/heatmap"
            )
            path = Path(dir_to_flow_frame, f"{str(frame_name).zfill(6)}.npy")
            if path.exists():
                frame = np.load(str(path))
            else:
                print(path)
                frame = frames[-1]
        return frame

    def _get_input(self, dir_to_img_frame, clip_start_frame):
        frame_names = [
            max(1, clip_start_frame + self.cfg.target_sampling_rate * i)
            for i in range(self.num_frames)
        ]
        frames = []
        for frame_name in frame_names:
            frame = self._get_frame(dir_to_img_frame, frame_name, self.mode, frames)
            frames.append(frame)

        # [T, H, W, C] -> [T*C, H, W] -> [C, T, H, W]
        if self.mode == "RGB":
            frames, _ = self.transform((frames, None))
            frames = frames.view((self.num_frames, 3) + frames.size()[-2:]).transpose(
                0, 1
            )
        elif self.mode == "flow":
            frames, _ = self.transform(frames)
            frames = frames.view((self.num_frames, 2) + frames.size()[-2:]).transpose(
                0, 1
            )
        elif self.mode == "pose":
            frames, _ = self.transform(frames)
            frames = frames.view((self.num_frames, 21) + frames.size()[-2:]).transpose(
                0, 1
            )

        # mask generator
        mask = self.mask_gen()

        return frames, mask

    def __getitem__(self, index):
        input = {}

        dir_to_img_frame = self._dir_to_img_frame[index]
        clip_start_frame = self._start_frame[index]
        action_label = self._action_label[index]
        action_idx = self._action_idx[index]

        # load frames
        frames, mask = self._get_input(dir_to_img_frame, clip_start_frame)

        input["frames"] = frames
        input["mask"] = mask
        input["action_label"] = action_label
        input["action_idx"] = action_idx

        return input, index

    def __len__(self):
        return len(self._start_frame)
