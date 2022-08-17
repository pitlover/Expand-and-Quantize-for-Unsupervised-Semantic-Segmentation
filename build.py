from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.dist_utils import is_distributed_set, get_rank, get_world_size
from model.quantizer import VectorQuantizer, EMAVectorQuantizer, EmbeddingEMA
from data.dataset import UnSegDataset


def split_params_for_optimizer(model):
    params = []
    params_no_wd = []
    for module_name, module in model.named_modules():
        if isinstance(module, (VectorQuantizer, EMAVectorQuantizer, EmbeddingEMA)):
            vq_params = [p for p in list(module.parameters(recurse=False)) if p.requires_grad]
            params_no_wd.extend(vq_params)
        else:
            for param in module.parameters(recurse=False):
                if not param.requires_grad:
                    continue

                if param.ndim > 1:
                    params.append(param)
                else:
                    params_no_wd.append(param)

    params_for_optimizer = [
        {"params": params},
        {"params": params_no_wd, "weight_decay": 0.0},
    ]
    return params_for_optimizer


def build_optimizer(cfg: dict, params):
    # cfg = cfg["optimizer"]["model" / "cluster" / "linear"]

    optimizer_type = cfg["name"].lower()
    if optimizer_type == "adam":
        optimizer = Adam(params,
                         lr=cfg["lr"],
                         betas=cfg.get("betas", (0.9, 0.999)),
                         weight_decay=cfg.get("weight_decay", 0.0))
    elif optimizer_type == "adamw":
        optimizer = AdamW(params,
                          lr=cfg["lr"],
                          betas=cfg.get("betas", (0.9, 0.999)),
                          weight_decay=cfg.get("weight_decay", 0.0))
    elif optimizer_type == "sgd":
        optimizer = SGD(params,
                        lr=cfg["lr"],
                        momentum=cfg.get("momentum", 0.9),
                        weight_decay=cfg.get("weight_decay", 0.0))
    else:
        raise ValueError(f"Unsupported optimizer type {optimizer_type}.")
    return optimizer


def build_scheduler(cfg: dict,
                    optimizer: SGD,
                    iter_per_epoch: int,
                    num_accum: int = 1
                    ):
    # cfg = cfg["scheduler"]["model" / "cluster" / "linear"]

    iter_per_epoch = iter_per_epoch // num_accum  # actual update count
    scheduler_type = cfg["name"].lower()
    if scheduler_type == "constant":
        scheduler = ConstantLR(optimizer,
                               factor=cfg.get("factor", 1.0),
                               total_iters=0)
    elif (scheduler_type == "cos") or (scheduler_type == "cosine"):
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=cfg["epochs"] * iter_per_epoch,
                                      eta_min=cfg.get("min_lr", 0.0))
    else:
        raise ValueError(f"Unsupported scheduler type {scheduler_type}.")
    return scheduler

    # warmup_cfg = cfg["warmup"]
    # warmup = LinearLR(
    #     optimizer,
    #     start_factor=warmup_cfg["start_factor"],
    #     end_factor=1.0,
    #     total_iters=warmup_cfg["epochs"] * iter_per_epoch,
    # )
    # decay_cfg = cfg["decay"]
    # decay = CosineAnnealingLR(
    #     optimizer,
    #     T_max=decay_cfg["epochs"] * iter_per_epoch,
    #     eta_min=decay_cfg["min_lr"],
    # )
    # scheduler = SequentialLR(
    #     optimizer,
    #     schedulers=[warmup, decay],
    #     milestones=[warmup_cfg["epochs"] * iter_per_epoch]
    # )


def build_dataset(cfg: dict, mode: str = "train") -> UnSegDataset:
    # cfg = cfg["dataset"]
    cfg = cfg[mode]

    dataset = UnSegDataset(
        mode=mode,
        data_dir=cfg["data_dir"],
        dataset_name=cfg["dataset_name"],
        crop_type=cfg["crop_type"],
        crop_ratio=cfg.get("crop_ratio", 0.5),
        loader_crop_type=cfg.get("loader_crop_type", "center"),
        res=cfg["res"]
    )
    return dataset


def build_dataloader(cfg: dict, dataset: UnSegDataset, mode: str = "train") -> DataLoader:
    # cfg = cfg["dataloader"]
    cfg = cfg[mode]

    shuffle = True if ("train" in mode) else False

    if not is_distributed_set():
        loader = DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            shuffle=shuffle,
            num_workers=cfg.get("num_workers", 1),
            pin_memory=True,
            drop_last=shuffle,
        )
    else:
        ddp_sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle,
            drop_last=shuffle,
        )
        world_size = get_world_size()
        loader = DataLoader(
            dataset,
            batch_size=max(cfg["batch_size"] // world_size, 1),
            num_workers=max((cfg["num_workers"] + world_size - 1) // world_size, 1),
            pin_memory=True,
            sampler=ddp_sampler
        )
    return loader
