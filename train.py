from typing import Dict, Tuple
from collections import OrderedDict
import os
import time
import pprint
import wandb
import torch
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.nn.utils.clip_grad import clip_grad_norm_

from utils.config_utils import prepare_config
from utils.wandb_utils import set_wandb
from utils.random_utils import set_seed
from utils.dist_utils import set_dist, is_distributed_set, is_master, barrier, get_world_size
from utils.dist_utils import all_reduce_dict
from utils.print_utils import time_log
from utils.param_utils import count_params, compute_param_norm

from build import build_dataset, build_dataloader, build_optimizer, build_scheduler, split_params_for_optimizer
from model.dino_unseg import DINOUnSeg
from model.metric import UnSegMetrics
from wrapper import DINOUnSegWrapper


def train_epoch(
        model: DINOUnSegWrapper,
        optimizers,
        schedulers,
        train_dataloader,
        valid_dataloader,
        cfg: Dict,
        device: torch.device,
        save_dir: str,
        current_epoch: int,
        current_iter: int,
        best_metric: Dict[str, float]
) -> Tuple[int, Dict[str, float]]:
    # cfg = cfg
    model_m = model.module if isinstance(model, DistributedDataParallel) else model
    model_m: DINOUnSegWrapper

    print_interval = cfg["train"]["print_interval_iters"]
    valid_interval = cfg["train"]["valid_interval_iters"]
    num_accum = cfg["train"].get("num_accum", 1)
    clip_grad = cfg["train"].get("clip_grad", 1.0)

    model.train()
    torch.cuda.empty_cache()
    torch.set_grad_enabled(True)  # same as 'with torch.enable_grad():'
    grad_norm = torch.tensor(0.0, dtype=torch.float32, device=device)  # placeholder

    forward_time = 0.0
    backward_time = 0.0
    step_time = 0.0

    best_iter = best_epoch = 0

    data_start_time = time.time()
    for it, data in enumerate(train_dataloader):

        s = time_log()
        s += f"Current iter: {current_iter} (epoch {current_epoch}, " \
             f"epoch done: {it / len(train_dataloader) * 100:.2f} %)\n"

        # -------------------------------- data -------------------------------- #
        img = data["img"].to(device, non_blocking=True)
        label = data["label"].to(device, non_blocking=True)
        data_time = time.time() - data_start_time

        # -------------------------------- loss -------------------------------- #
        if it % num_accum == 0:
            for optim in optimizers:
                optim.zero_grad(set_to_none=True)

        if it % num_accum == (num_accum - 1):  # update step
            forward_start_time = time.time()
            total_loss, output, _ = model(img, label)  # total_loss, output, (linear_preds, cluster_preds)
            forward_time = time.time() - forward_start_time

            backward_start_time = time.time()
            loss = total_loss / num_accum
            loss.backward()
            backward_time = time.time() - backward_start_time

            step_start_time = time.time()
            grad_norm = clip_grad_norm_(model_m.model.parameters(), max_norm=clip_grad)
            for optim, sched in zip(optimizers, schedulers):
                optim.step()
                sched.step()
            step_time = time.time() - step_start_time

            current_iter += 1
            model_m.model.restart()

        elif isinstance(model, DistributedDataParallel):  # non-update step and DDP
            with model.no_sync():
                total_loss, output, _ = model(img, label)  # total_loss, output, (linear_preds, cluster_preds)

                loss = total_loss / num_accum
                loss.backward()
        else:  # non-update step and not DDP
            total_loss, output, _ = model(img, label)  # total_loss, output, (linear_preds, cluster_preds)

            loss = total_loss / num_accum
            loss.backward()

        # -------------------------------- print -------------------------------- #

        if (it > 0) and (it % print_interval == 0):
            output = all_reduce_dict(output, op="mean")
            param_norm = compute_param_norm(model_m.model.parameters())
            lr = schedulers[0].get_last_lr()[0]

            for k, v in output.items():
                s += f"... {k}: {v.item() if isinstance(v, torch.Tensor) else v:.6f}\n"
            s += f"... LR: {lr:.6f}\n"
            s += f"... grad/param norm: {grad_norm.item():.3f} / {param_norm.item():.3f}\n"
            s += f"... batch_size x num_accum x gpus = " \
                 f"{int(label.shape[0])} x {num_accum} x {get_world_size()}\n"
            s += f"... data/fwd/bwd/step time: " \
                 f"{data_time:.3f} / {forward_time:.3f} / {backward_time:.3f} / {step_time:.3f}"

            if is_master():
                print(s)
                log_dict = {
                    "grad_norm": grad_norm.item(),
                    "param_norm": param_norm.item(),
                    "lr": lr,
                    "iterations": current_iter,
                }
                for k, v in output.items():
                    log_dict[k] = v.item() if isinstance(v, torch.Tensor) else v
                wandb.log(log_dict)

        if (it > 0) and (it % valid_interval == 0):
            _, cluster_result, linear_result = valid_epoch(
                model, valid_dataloader, cfg, device, current_iter, is_crf=False)

            if is_master():
                if best_metric["Cluster_mIoU"] <= cluster_result["iou"].item():
                    s = time_log()
                    s += f"Valid updated!\n"
                    s += f"... Cluster mIou: {best_metric['Cluster_mIoU']} ->  {cluster_result['iou'].item()}\n"
                    s += f"... Cluster Accuracy: {best_metric['Cluster_Accuracy']} ->  {cluster_result['accuracy'].item()}\n"
                    s += f"... Linear mIou: {best_metric['Linear_mIoU']} ->  {linear_result['iou'].item()}\n"
                    s += f"... Linear Accuracy: {best_metric['Linear_Accuracy']} ->  {linear_result['accuracy'].item()}"
                    print(s)

                    best_iter = current_iter
                    best_epoch = current_epoch
                    best_metric["Cluster_mIoU"] = cluster_result["iou"].item()
                    best_metric["Cluster_Accuracy"] = cluster_result["accuracy"].item()
                    best_metric["Linear_mIoU"] = linear_result["iou"].item()
                    best_metric["Linear_Accuracy"] = linear_result["accuracy"].item()
                    torch.save({
                        "model": model_m.state_dict(),
                        "optimizer": [optim.state_dict() for optim in optimizers],
                        "scheduler": [sched.state_dict() for sched in schedulers],
                        "best": best_metric.copy(),
                        "epoch": current_epoch,
                        "iter": current_iter,
                    }, os.path.join(save_dir, "best.pth"))
                else:
                    s = time_log()
                    s += f"Valid NOT updated ...\n"
                    s += f"... previous best was at {best_epoch} epoch, {best_iter} iters\n"
                    s += f"... Cluster mIou: {best_metric['Cluster_mIoU']} (best) vs {cluster_result['iou'].item()}\n"
                    s += f"... Cluster Accuracy: {best_metric['Cluster_Accuracy']} (best) vs {cluster_result['accuracy'].item()}\n"
                    s += f"... Linear mIou: {best_metric['Linear_mIoU']} (best) vs {linear_result['iou'].item()}\n"
                    s += f"... Linear Accuracy: {best_metric['Linear_Accuracy']} (best) vs {linear_result['accuracy'].item()}"
                    print(s)

            model.train()
            torch.set_grad_enabled(True)  # same as 'with torch.enable_grad():'

        data_start_time = time.time()

    return current_iter, best_metric


def valid_epoch(
        model: DINOUnSegWrapper,
        dataloader,
        cfg: Dict,
        device: torch.device,
        current_iter: int,
        is_crf: bool = False,
) -> Tuple[Dict, Dict, Dict]:
    # model_m = model.module if isinstance(model, DistributedDataParallel) else model
    # model_m: DINOUnSegWrapper

    cluster_m = UnSegMetrics(cfg["num_classes"], extra_classes=cfg["eval"]["extra_classes"], compute_hungarian=True)
    linear_m = UnSegMetrics(cfg["num_classes"], extra_classes=0, compute_hungarian=False)

    cluster_m.reset()
    linear_m.reset()

    model.eval()
    torch.cuda.empty_cache()
    torch.set_grad_enabled(False)  # same as 'with torch.no_grad():'

    s = time_log()
    s += f"Valid iter: {current_iter}\n"

    valid_start_time = time.time()
    result = dict()
    count = 0
    for it, data in enumerate(dataloader):
        # -------------------------------- data -------------------------------- #
        img = data["img"].to(device, non_blocking=True)
        label = data["label"].to(device, non_blocking=True)

        # -------------------------------- loss -------------------------------- #
        _, output, (linear_preds, cluster_preds) = model(img, label, is_crf=is_crf)
        cluster_m.update(cluster_preds, label)
        linear_m.update(linear_preds, label)

        for k, v in output.items():
            if k not in result:
                result[k] = v
            else:
                result[k] += v
        count += 1

    barrier()
    cluster_result = cluster_m.compute()  # {iou, accuracy}
    linear_result = linear_m.compute()  # {iou, accuracy}

    barrier()
    for k, v in result.items():
        result[k] /= count
    result = all_reduce_dict(result, op="mean")

    valid_time = time.time() - valid_start_time

    if is_master():
        s += f"Cluster: mIoU {cluster_result['iou'].item():.6f}, acc: {cluster_result['accuracy'].item():.6f}\n"
        s += f"Linear: mIoU {linear_result['iou'].item():.6f}, acc: {linear_result['accuracy'].item():.6f}\n"
        for k, v in result.items():
            s += f"... {k}: {v.item() if isinstance(v, torch.Tensor) else v:.6f}\n"
        s += f"... time: {valid_time:.3f} sec"

        print(s)
        log_dict = {
            "iterations": current_iter,
            "Cluster_mIoU": cluster_result['iou'].item(),
            "Cluster_Accuracy": cluster_result['accuracy'].item(),
            "Linear_mIoU": linear_result['iou'].item(),
            "Linear_Accuracy": linear_result['accuracy'].item(),
        }
        for k, v in result.items():
            log_dict[k] = v.item() if isinstance(v, torch.Tensor) else v
        wandb.log(log_dict)

    return result, cluster_result, linear_result


def run(cfg: Dict, debug: bool = False) -> None:
    # ======================================================================================== #
    # Initialize
    # ======================================================================================== #
    device, local_rank = set_dist(device_type="cuda")
    if is_master():
        pprint.pprint(cfg)  # print config to check if all arguments are correctly given.

    save_dir = set_wandb(cfg, force_mode="disabled" if debug else None)
    set_seed(seed=cfg["seed"] + local_rank)

    # ======================================================================================== #
    # Data
    # ======================================================================================== #
    train_dataset = build_dataset(cfg["dataset"], mode="train")
    train_dataloader = build_dataloader(cfg["dataloader"], train_dataset, mode="train")

    valid_dataset = build_dataset(cfg["dataset"], mode="val")
    valid_dataloader = build_dataloader(cfg["dataloader"], valid_dataset, mode="val")

    # ======================================================================================== #
    # Model
    # ======================================================================================== #
    model = DINOUnSeg(cfg["model"])
    model = DINOUnSegWrapper(cfg, model)
    model = model.to(device)
    if is_distributed_set():
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=device)
        model_m = model.module  # actual model without wrapping
    else:
        model_m = model

    if is_master():
        print(model)
        p1, p2 = count_params(model_m.model.parameters(), requires_grad=True)
        print(f"Model parameters: {p1} tensors, {p2} elements.")

    if cfg["resume"]["checkpoint"] is not None:
        raise NotImplementedError  # resume
    else:
        ckpt = None

    # ======================================================================================== #
    # Optimizer & Scheduler
    # ======================================================================================== #
    model_params = split_params_for_optimizer(model_m.model)
    cluster_params = model_m.evaluator.cluster_probe.parameters()
    linear_params = model_m.evaluator.linear_probe.parameters()

    model_optimizer = build_optimizer(cfg["optimizer"]["model"], model_params)
    cluster_optimizer = build_optimizer(cfg["optimizer"]["cluster"], cluster_params)
    linear_optimizer = build_optimizer(cfg["optimizer"]["linear"], linear_params)

    optimizers = [model_optimizer, cluster_optimizer, linear_optimizer]

    iter_per_epoch = len(train_dataloader)
    num_accum = cfg["train"].get("num_accum", 1)
    model_scheduler = build_scheduler(cfg["scheduler"]["model"], model_optimizer, iter_per_epoch, num_accum)
    cluster_scheduler = build_scheduler(cfg["scheduler"]["cluster"], cluster_optimizer, iter_per_epoch, num_accum)
    linear_scheduler = build_scheduler(cfg["scheduler"]["linear"], linear_optimizer, iter_per_epoch, num_accum)

    schedulers = [model_scheduler, cluster_scheduler, linear_scheduler]

    if ckpt is not None:
        raise NotImplementedError  # resume

    # ======================================================================================== #
    # Trainer
    # ======================================================================================== #

    # -------- config -------- #
    train_cfg = cfg["train"]
    max_epochs = train_cfg["max_epochs"]

    # -------- status -------- #
    best_metric = dict()
    best_metric["Cluster_mIoU"] = 0.0
    best_metric["Cluster_Accuracy"] = 0.0
    best_metric["Linear_mIoU"] = 0.0
    best_metric["Linear_Accuracy"] = 0.0

    if ckpt is not None:
        raise NotImplementedError  # resume

    current_epoch = 0
    current_iter = 0

    # -------- main loop -------- #
    while current_epoch < max_epochs:
        if is_master():
            s = time_log()
            s += f"Start train epoch {current_epoch} / {max_epochs} (iter: {current_iter})"
            print(s)

        if is_distributed_set():
            # reset random seed of sampler, sampler should be DistributedSampler.
            train_dataloader.sampler.set_epoch(current_epoch)  # noqa

        # -------- train body -------- #
        epoch_start_time = time.time()  # second
        current_iter, best_metric = train_epoch(model, optimizers, schedulers, train_dataloader, valid_dataloader,
                                                cfg, device, save_dir, current_epoch, current_iter, best_metric)
        epoch_time = time.time() - epoch_start_time
        if is_master():
            s = time_log()
            s += f"End train epoch {current_epoch} / {max_epochs}\n"
            s += f"... time: {epoch_time:.3f} sec"
            print(s)

        barrier()
        current_epoch += 1

    # -------- final evaluation -------- #
    # TODO load best model

    final_start_time = time.time()
    s = time_log()
    s += "Final evaluation (before CRF)\n"

    _, cluster_result, linear_result = valid_epoch(model, valid_dataloader, cfg, device, current_iter, is_crf=False)
    s += f"Cluster: mIoU {cluster_result['iou'].item():.6f}, acc: {cluster_result['accuracy'].item():.6f}\n"
    s += f"Linear: mIoU {linear_result['iou'].item():.6f}, acc: {linear_result['accuracy'].item():.6f}\n"

    s += "Final evaluation (after CRF)\n"
    _, cluster_result, linear_result = valid_epoch(model, valid_dataloader, cfg, device, current_iter, is_crf=True)
    s += f"Cluster: mIoU {cluster_result['iou'].item():.6f}, acc: {cluster_result['accuracy'].item():.6f}\n"
    s += f"Linear: mIoU {linear_result['iou'].item():.6f}, acc: {linear_result['accuracy'].item():.6f}\n"

    final_time = time.time() - final_start_time
    s += f"... time: {final_time:.3f} sec"

    if is_master():
        wandb.finish()


if __name__ == '__main__':
    args, config = prepare_config()
    run(config, args.debug)
