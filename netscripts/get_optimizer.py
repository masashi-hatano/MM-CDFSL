import json

from torch import optim as optim
from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nadam import Nadam
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP
from timm.scheduler import CosineLRScheduler


def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split(".")[1])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))


def get_parameter_groups(
    model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None
):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.0
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.0

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def create_optimizer(
    cfg,
    model,
    get_num_layer=None,
    get_layer_scale=None,
    filter_bias_and_bn=True,
    skip_list=None,
):
    opt_lower = cfg.opt.lower()
    weight_decay = cfg.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, "no_weight_decay"):
            skip = model.no_weight_decay()
        parameters = get_parameter_groups(
            model, weight_decay, skip, get_num_layer, get_layer_scale
        )
        weight_decay = 0.0
    else:
        parameters = model.parameters()

    opt_args = dict(lr=cfg.lr, weight_decay=weight_decay)
    if hasattr(cfg, "opt_eps") and cfg.opt_eps is not None:
        opt_args["eps"] = cfg.opt_eps
    if hasattr(cfg, "opt_betas") and cfg.opt_betas is not None:
        opt_args["betas"] = cfg.opt_betas

    # print("optimizer settings:", opt_args)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "sgd" or opt_lower == "nesterov":
        opt_args.pop("eps", None)
        optimizer = optim.SGD(
            parameters, momentum=cfg.momentum, nesterov=True, **opt_args
        )
    elif opt_lower == "momentum":
        opt_args.pop("eps", None)
        optimizer = optim.SGD(
            parameters, momentum=cfg.momentum, nesterov=False, **opt_args
        )
    elif opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "nadam":
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == "adamp":
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == "sgdp":
        optimizer = SGDP(parameters, momentum=cfg.momentum, nesterov=True, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "adafactor":
        if not cfg.lr:
            opt_args["lr"] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == "adahessian":
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == "rmsprop":
        optimizer = optim.RMSprop(
            parameters, alpha=0.9, momentum=cfg.momentum, **opt_args
        )
    elif opt_lower == "rmsproptf":
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=cfg.momentum, **opt_args)
    elif opt_lower == "nvnovograd":
        optimizer = NvNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == "lookahead":
            optimizer = Lookahead(optimizer)

    return optimizer


def get_optimizer(cfg, model, niter_per_epoch):
    optimizer = create_optimizer(cfg, model)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=cfg.epochs * niter_per_epoch,
        lr_min=cfg.min_lr,
        warmup_t=cfg.warmup_epochs * niter_per_epoch,
        warmup_lr_init=cfg.warmup_lr,
        warmup_prefix=True,
    )

    scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

    return optimizer, scheduler


def get_optimizer_da(cfg, model, dann, niter_per_epoch):
    optim_params = [{"params": model.parameters()}, {"params": dann.parameters()}]
    optimizer = optim.AdamW(optim_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    # optimizer = create_optimizer(cfg, model)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=cfg.epochs * niter_per_epoch,
        lr_min=cfg.min_lr,
        warmup_t=cfg.warmup_epochs * niter_per_epoch,
        warmup_lr_init=cfg.warmup_lr,
        warmup_prefix=True,
    )

    scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

    return optimizer, scheduler


def get_optimizer_sdt(cfg, model, niter_per_epoch):
    optim_params = [{"params": m.parameters()} for m in model]
    optimizer = optim.AdamW(optim_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    # optimizer = create_optimizer(cfg, model)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=cfg.epochs * niter_per_epoch,
        lr_min=cfg.min_lr,
        warmup_t=cfg.warmup_epochs * niter_per_epoch,
        warmup_lr_init=cfg.warmup_lr,
        warmup_prefix=True,
    )

    scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

    return optimizer, scheduler


def get_optimizer_weight(cfg, model, niter_per_epoch):
    optim_params = [{"params": model.parameters()}]
    optimizer = optim.AdamW(optim_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    # optimizer = create_optimizer(cfg, model)
    # scheduler = CosineLRScheduler(
    #     optimizer, t_initial=cfg.epochs*niter_per_epoch, lr_min=cfg.min_lr,
    #     warmup_t=cfg.warmup_epochs*niter_per_epoch, warmup_lr_init=cfg.warmup_lr, warmup_prefix=True)

    # scheduler = {
    #     "scheduler": scheduler,
    #     "interval": "step",
    #     "frequency": 1
    # }

    return optimizer


def get_optimizer_mmdistill(cfg, model, niter_per_epoch):
    optim_params = [{"params": m.parameters()} for m in model]
    optimizer = optim.AdamW(optim_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    # optimizer = create_optimizer(cfg, model)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=cfg.epochs * niter_per_epoch,
        lr_min=cfg.min_lr,
        warmup_t=cfg.warmup_epochs * niter_per_epoch,
        warmup_lr_init=cfg.warmup_lr,
        warmup_prefix=True,
    )

    scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

    return optimizer, scheduler
