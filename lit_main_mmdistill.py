import logging
import os
import random
import warnings

import hydra
import numpy as np
import torch
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodule.lit_unlabel_combined_mm_data_module import UnlabelCombinedMMDataModule
from models.lit_MMDistillTrainer import MMDistillTrainer

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config_mmdistill.yaml")
def main(cfg):
    # initialize random seeds
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # data module
    data_module = UnlabelCombinedMMDataModule(cfg)

    # model
    model = MMDistillTrainer(cfg)

    if torch.cuda.is_available() and len(cfg.devices):
        print(f"Using {len(cfg.devices)} GPUs !")

    train_logger = loggers.TensorBoardLogger("tensor_board", default_hp_metric=False)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_top1_action",
        dirpath="checkpoints/",
        filename="{epoch:02d}-{val_top1_action:.4f}",
        save_top_k=5,
        mode="max",
    )

    trainer = Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        strategy=cfg.strategy,
        max_epochs=cfg.trainer.epochs,
        logger=train_logger,
        callbacks=[checkpoint_callback],
        detect_anomaly=True,
        use_distributed_sampler=False,
        check_val_every_n_epoch=5,
    )

    if cfg.train:
        trainer.fit(model, data_module)
        print(trainer.callback_metrics)

    if cfg.test:
        logging.basicConfig(level=logging.DEBUG)
        trainer.test(model, data_module)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
