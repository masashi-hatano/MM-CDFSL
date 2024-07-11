import logging

# from models.modeling_finetune import *
from models.modeling_pretrain import (
    pretrain_videomae_base_patch16_224,
    pretrain_videomae_huge_patch16_224,
    pretrain_videomae_large_patch16_224,
    pretrain_videomae_small_patch16_224,
)
from models.videomae_classifier import videomae_classifier_small_patch16_224

logger = logging.getLogger(__name__)


def get_model(cfg, ckpt_pth=None, input_size=224, patch_size=16, in_chans=None):
    print(f"Creating model: {cfg.trainer.model}")
    if cfg.trainer.model.split("_")[0] == "pretrain":
        scale = cfg.trainer.model.split("_")[2]
        if scale == "small":
            func = pretrain_videomae_small_patch16_224
        elif scale == "base":
            func = pretrain_videomae_base_patch16_224
        elif scale == "large":
            func = pretrain_videomae_large_patch16_224
        elif scale == "huge":
            func = pretrain_videomae_huge_patch16_224
        else:
            raise Exception(f"{scale} is not supported!")
        if cfg.data_module.modality.mode == "RGB":
            assert cfg.trainer.modality.in_chans == 3
            assert cfg.trainer.pretrain is not None
        elif cfg.data_module.modality.mode == "flow":
            assert cfg.trainer.modality.in_chans == 2
            assert cfg.trainer.pretrain is not None
        elif cfg.data_module.modality.mode == "pose":
            assert cfg.trainer.modality.in_chans == 21
            assert cfg.trainer.pretrain is not None
        else:
            raise Exception(f"{cfg.data_module.modality.mode} is not supported!")

        model = func(
            ckpt_pth=cfg.trainer.pretrain,
            img_size=cfg.data_module.modality.input_size,
            patch_size=cfg.data_module.modality.patch_size[0],
            in_chans=cfg.trainer.modality.in_chans,
            decoder_num_classes=cfg.trainer.modality.decoder_num_classes,
        )
    elif cfg.trainer.model.split("_")[1] == "classifier":
        scale = cfg.trainer.model.split("_")[2]
        if scale == "small":
            func = videomae_classifier_small_patch16_224
        else:
            raise Exception(f"{scale} is not supported!")

        model = func(
            ckpt_pth=ckpt_pth,
            img_size=input_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes_action=cfg.data_module.num_classes_action,
            use_mean_pooling=cfg.trainer.use_mean_pooling,
        )
    return model
