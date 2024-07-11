import logging

from datamodule.dataset.epic_fewshot_eval_dataset import EPICFewshotEvalDataset
from datamodule.dataset.meccano_fewshot_eval_dataset import MECCANOFewshotEvalDataset
from datamodule.dataset.wear_fewshot_eval_dataset import WEARFewshotEvalDataset

logger = logging.getLogger(__name__)


def get_fewshot_eval_dataset(cfg, transform, mask_gen, num_frames, mode):
    if cfg.target_dataset.lower() == "epic":
        fewshot_eval_dataset = EPICFewshotEvalDataset(
            cfg, transform, mask_gen, num_frames, mode
        )
    elif cfg.target_dataset.lower() == "meccano":
        fewshot_eval_dataset = MECCANOFewshotEvalDataset(
            cfg, transform, mask_gen, num_frames, mode
        )
    elif cfg.target_dataset.lower() == "wear":
        fewshot_eval_dataset = WEARFewshotEvalDataset(
            cfg, transform, mask_gen, num_frames, mode
        )
    else:
        raise Exception(f"{cfg.target_dataset} is not supported!")
    logger.info(f"Using {cfg.target_dataset} as Fewshot Evaluation Dataset")
    return fewshot_eval_dataset
