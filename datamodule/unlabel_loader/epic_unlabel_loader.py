from pathlib import Path
import json
import logging


logger = logging.getLogger(__name__)


class EPICUnlabelLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self._construct_loader(cfg)

    def _construct_loader(self, cfg):
        # initialization
        self._dir_to_img_frame = []
        self._uid = []
        self._participant_id = []
        self._video_id = []
        self._start_frame = []
        self._stop_frame = []

        # read annotation json file
        with open(cfg.unlabel_json_path) as f:
            data = json.load(f)
        assert data["split"] == "train (unlabel)"

        for i, clip_dict in enumerate(data["clips"]):
            uid = clip_dict["uid"]
            participant_id = clip_dict["participant_id"]
            video_id = clip_dict["video_id"]
            start_frame = int(clip_dict["start_frame"])

            dir_to_img_frame = Path(
                cfg.target_data_dir,
                "frames/rgb/train",
                participant_id,
                video_id,
            )
            self._dir_to_img_frame.append(dir_to_img_frame)
            self._uid.append(uid)
            self._start_frame.append(start_frame)

        logger.info(f"Constructing EPIC unlabel dataloader (size: {len(self._uid)})")

    def get_frame_str(self, frame_name):
        return f"frame_{str(frame_name).zfill(10)}.jpg"

    def __len__(self):
        return len(self._uid)
