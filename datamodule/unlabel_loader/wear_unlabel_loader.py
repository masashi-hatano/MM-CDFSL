import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class WEARUnlabelLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self._construct_loader(cfg)

    def _construct_loader(self, cfg):
        # initialization
        self._dir_to_img_frame = []
        self._start_frame = []

        # read annotation json file
        with open(cfg.unlabel_json_path) as f:
            data = json.load(f)
        assert data["split"] == "train (unlabel)"

        for i, clip_dict in enumerate(data["clips"]):
            video_id = clip_dict["video_id"]
            start_frame = int(clip_dict["start_frame"])

            dir_to_img_frame = Path(
                cfg.target_data_dir,
                "RGB_frames",
                video_id,
            )
            self._dir_to_img_frame.append(dir_to_img_frame)
            self._start_frame.append(start_frame)

        logger.info(
            f"Constructing WEAR unlabel dataloader (size: {len(self._start_frame)})"
        )

    def get_frame_str(self, frame_name):
        return f"{str(frame_name).zfill(6)}.jpg"

    def __len__(self):
        return len(self._start_frame)
