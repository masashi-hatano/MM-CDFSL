from torchvision import transforms

from datamodule.utils.masking_generator import (
    RandomMaskingGenerator,
    TubeMaskingGenerator,
)
from datamodule.utils.transform import (
    GroupMultiScaleCrop,
    GroupNormalize,
    GroupScale,
    Stack,
    ToTensor,
    ToTorchFormatTensor,
)


class DataAugmentationForVideoMAERGB(object):
    def __init__(
        self,
        cfg,
        num_frames=16,
        input_size=224,
        patch_size=[16, 16],
        mean=[0.485, 0.456, 0.406],  # IMAGENET_DEFAULT_MEAN
        std=[0.229, 0.224, 0.225],  # IMAGENET_DEFAULT_STD
        multi_scale_crop=True,
    ):
        aug_list = []

        if multi_scale_crop:
            aug_list.append(GroupMultiScaleCrop(input_size, [1, 0.875, 0.75, 0.66]))
        else:
            aug_list.append(GroupScale(size=(input_size, input_size)))
        aug_list.append(Stack(roll=False))
        aug_list.append(ToTorchFormatTensor(div=True))
        aug_list.append(GroupNormalize(mean, std))
        self.transform = transforms.Compose(aug_list)

        window_size = (
            num_frames // 2,
            input_size // patch_size[0],
            input_size // patch_size[1],
        )

        if cfg.mask_type == "tube":
            self.masked_position_generator = TubeMaskingGenerator(
                window_size, cfg.mask_ratio
            )
        else:
            self.masked_position_generator = lambda: None

    def __call__(self, images):
        process_data, _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(
            self.masked_position_generator
        )
        repr += ")"
        return repr


class MaskGeneration(object):
    def __init__(
        self,
        cfg,
        num_frames=16,
        input_size=224,
        patch_size=[16, 16],
    ):
        window_size = (
            num_frames // 2,
            input_size // patch_size[0],
            input_size // patch_size[1],
        )
        if cfg.mask_type == "tube":
            self.masked_position_generator = TubeMaskingGenerator(
                window_size, cfg.mask_ratio
            )
        elif cfg.mask_type == "random":
            self.masked_position_generator = RandomMaskingGenerator(
                window_size, cfg.mask_ratio
            )
        else:
            self.masked_position_generator = lambda: None

    def __call__(self):
        return self.masked_position_generator()


class DataAugmentationForUnlabelRGB(object):
    def __init__(
        self, cfg, input_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ):
        self.cfg = cfg
        self.input_size = input_size
        self.mean = mean
        self.std = std
        self._construct_no_aug()
        self._construct_weak_aug()
        self._construct_strong_aug()

    def _construct_no_aug(self):
        self.no_aug = transforms.Compose(
            [
                ToTensor(),
                transforms.Resize((self.input_size, self.input_size)),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def _construct_weak_aug(self):
        self.weak_aug = transforms.Compose(
            [
                ToTensor(),
                transforms.RandomResizedCrop(self.input_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def _construct_strong_aug(self):
        self.strong_aug = transforms.Compose(
            [
                ToTensor(),
                transforms.RandomResizedCrop(self.input_size),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )


class DataAugmentationForVideoMAEMM(object):
    def __init__(self, cfg, mean, std):
        self.mean = mean
        self.std = std
        self.transform = transforms.Compose(
            [ToTensor(), transforms.Normalize(mean=self.mean, std=self.std)]
        )

        if cfg.mask_type == "tube":
            window_size = (
                cfg.num_frames // 2,
                cfg.input_size // cfg.patch_size[0],
                cfg.input_size // cfg.patch_size[1],
            )
            self.masked_position_generator = TubeMaskingGenerator(
                window_size, cfg.mask_ratio
            )
        else:
            self.masked_position_generator = lambda: None

    def __call__(self, images):
        process_data = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(
            self.masked_position_generator
        )
        repr += ")"
        return repr


class DataAugmentationForUnlabelMM(object):
    def __init__(self, cfg, mean, std):
        self.cfg = cfg
        self.mean = mean
        self.std = std
        self._construct_weak_aug()
        self._construct_strong_aug()

    def _construct_weak_aug(self):
        self.weak_aug = transforms.Compose(
            [
                ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def _construct_strong_aug(self):
        self.strong_aug = transforms.Compose(
            [
                ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
                # GaussianNoise(variance=self.variance),
            ]
        )
