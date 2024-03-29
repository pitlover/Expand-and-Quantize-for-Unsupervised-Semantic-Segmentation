from typing import Dict, Tuple
import os
import time
import pprint
import wandb
import torch
import torch.nn as nn
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.nn.utils.clip_grad import clip_grad_norm_

from utils.config_utils import prepare_config
from utils.wandb_utils import set_wandb
from utils.random_utils import set_seed
from utils.dist_utils import set_dist, is_distributed_set, is_master, barrier, get_world_size
from utils.dist_utils import all_reduce_dict
from utils.print_utils import time_log
from utils.param_utils import count_params, compute_param_norm
from utils.visualize_utils import visualization

from build import build_dataset, build_dataloader, build_model, build_optimizer, build_scheduler, \
    split_params_for_optimizer
from model.metric import UnSegMetrics
from wrapper.UnsegWrapper import DINOUnSegWrapper
from collections import defaultdict


def train_epoch(
        model,
        optimizers,
        schedulers,
        train_dataloader,
        valid_dataloader,
        cfg: Dict,
        device: torch.device,
        save_dir: str,
        current_epoch: int,
        current_iter: int,
        best_metric: Dict[str, float],
        best_epoch: int,
        best_iter: int,
        scaler: torch.cuda.amp.GradScaler,
) -> Tuple[int, Dict[str, float], int, int]:
    # cfg = cfg
    model_m = model.module if isinstance(model, DistributedDataParallel) else model
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

    data_start_time = time.time()

    for it, data in enumerate(train_dataloader):
        s = time_log()
        s += f"Current iter: {it} (epoch {current_epoch}, " \
             f"epoch done: {it / len(train_dataloader) * 100:.2f} %)\n"
        # -------------------------------- data -------------------------------- #
        img = data["img"].to(device, non_blocking=True)
        aug_img = data["aug_img"].to(device, non_blocking=True)
        img_pos = data["img_pos"].to(device, non_blocking=True)
        img_path = data["img_path"]
        label = data["label"].to(device, non_blocking=True)

        data_time = time.time() - data_start_time
        # -------------------------------- loss -------------------------------- #
        if it % num_accum == 0:
            for optim in optimizers:
                optim.zero_grad(set_to_none=True)

        if it % num_accum == (num_accum - 1):  # update step
            forward_start_time = time.time()
            with torch.cuda.amp.autocast(enabled=True):
                total_loss, output, linear_pred = model(img=img, aug_img=aug_img, label=label, img_pos=img_pos,
                                                        img_path=img_path,
                                                        it=it)  # total_loss, output, (linear_preds, cluster_preds)
            forward_time = time.time() - forward_start_time
            backward_start_time = time.time()
            loss = total_loss / num_accum
            scaler.scale(loss).backward()
            backward_time = time.time() - backward_start_time

            step_start_time = time.time()
            scaler.unscale_(optimizers[0])
            grad_norm = clip_grad_norm_(model_m.model.parameters(), max_norm=clip_grad)

            for optim, sched in zip(optimizers, schedulers):
                scaler.step(optim)
                scaler.update()
                sched.step()
            step_time = time.time() - step_start_time

            current_iter += 1

            # for n, p in model.named_parameters():
            #     if p.grad is None:
            #         print(f'{n} has no grad')
            # exit()

        elif isinstance(model, DistributedDataParallel):  # non-update step and DDP
            with model.no_sync():
                with torch.cuda.amp.autocast(enabled=True):
                    total_loss, output, linear_pred = model(img=img, aug_img=aug_img, img_pos=img_pos, label=label
                                                            )  # total_loss, output, (linear_preds, cluster_preds)
                loss = total_loss / num_accum
                # loss.backward()
                scaler.scale(loss).backward()

        else:  # non-update step
            # and not DDP
            with torch.cuda.amp.autocast(enabled=True):
                total_loss, output, linear_pred = model(img=img, aug_img=aug_img, img_pos=img_pos,
                                                        label=label)  # total_loss, output, (linear_preds, cluster_preds)

            loss = total_loss / num_accum
            # loss.backward()
            scaler.scale(loss).backward()

        # -------------------------------- print -------------------------------- #

        if it % print_interval == 0:
            output = all_reduce_dict(output, op="mean")
            param_norm = compute_param_norm(model_m.model.parameters())
            lr = schedulers[0].get_last_lr()[0]
            for k, v in output.items():
                s += f"... {k}: {v.item() if isinstance(v, torch.Tensor) else v:.6f}\n"
            s += f"... LR: {lr:.6f}\n"
            s += f"... grad/param norm: {grad_norm.item():.3f} / {param_norm.item():.3f}\n"
            s += f"... batch_size x num_accum x gpus = " \
                 f"{int(label.shape[0])} x {num_accum} x {get_world_size()} = {int(label.shape[0]) * num_accum * get_world_size()}\n"
            s += f"... data/fwd/bwd/step time: " \
                 f"{data_time:.3f} / {forward_time:.3f} / {backward_time:.3f} / {step_time:.3f}"

            if is_master():
                print(s)
                log_dict = {
                    "grad_norm": grad_norm.item(),
                    "param_norm": param_norm.item(),
                    "lr": lr,
                    "iters": current_iter,
                }
                for k, v in output.items():
                    log_dict[k] = v.item() if isinstance(v, torch.Tensor) else v
                wandb.log(log_dict)

        if (it > 0) and (it % valid_interval == 0):
            _, linear_result = valid_epoch(
                model, valid_dataloader, cfg, device, current_iter, is_crf=False)

            if is_master():
                if best_metric["Linear_mIoU"] <= linear_result["iou"].item():
                    s = time_log()
                    s += f"Valid updated!\n"
                    s += f"... previous best was at {best_epoch} epoch, {best_iter} iters\n"
                    s += f"... Linear mIoU: {best_metric['Linear_mIoU']:.6f} ->  {linear_result['iou'].item():.6f}\n"
                    s += f"... Linear Accuracy: {best_metric['Linear_Accuracy']:.6f} ->  {linear_result['accuracy'].item():.6f}"
                    print(s)

                    best_iter = current_iter
                    best_epoch = current_epoch
                    best_metric["Linear_mIoU"] = linear_result["iou"].item()
                    best_metric["Linear_Accuracy"] = linear_result["accuracy"].item()
                    torch.save({
                        "model": model_m.state_dict(),
                        "optimizer": [optim.state_dict() for optim in optimizers],
                        "scheduler": [sched.state_dict() for sched in schedulers],
                        "best": best_metric.copy(),
                        "epoch": current_epoch,
                        "iter": current_iter,
                        "scaler": scaler.state_dict(),
                    }, os.path.join(save_dir, "best.pth"))
                else:
                    s = time_log()
                    s += f"Valid NOT updated ...\n"
                    s += f"... previous best was at {best_epoch} epoch, {best_iter} iters\n"
                    s += f"... Linear mIoU: {best_metric['Linear_mIoU']:.6f} (best) vs {linear_result['iou'].item():.6f}\n"
                    s += f"... Linear Accuracy: {best_metric['Linear_Accuracy']:.6f} (best) vs {linear_result['accuracy'].item():.6f}"
                    print(s)

            model.train()
            torch.cuda.empty_cache()
            torch.set_grad_enabled(True)  # same as 'with torch.enable_grad():'

        data_start_time = time.time()
    return current_iter, best_metric, best_epoch, best_iter


def valid_epoch(
        model: DINOUnSegWrapper,
        dataloader,
        cfg: Dict,
        device: torch.device,
        current_iter: int,
        is_crf: bool = False
) -> Tuple[Dict, Dict]:
    # model_m = model.module if isinstance(model, DistributedDataParallel) else model
    # model_m: DINOUnSegWrapper

    linear_m = UnSegMetrics(cfg["num_classes"], extra_classes=0, compute_hungarian=False, device=device)

    linear_m.reset()

    model.eval()
    torch.cuda.empty_cache()
    torch.set_grad_enabled(False)  # same as 'with torch.no_grad():'

    s = time_log()
    s += f"Valid iter: {current_iter}\n"

    valid_start_time = time.time()
    result = dict()
    count = 0
    saved_data = defaultdict(list)

    if cfg["is_visualize"]:
        os.makedirs(cfg["visualize_path"], exist_ok=True)

    # from npy_append_array import NpyAppendArray
    #
    # pq_num = "pq_16"
    # f_data = NpyAppendArray(f'./index/{pq_num}/data_1.npy')
    # f_label = NpyAppendArray(f'./index/{pq_num}/label_1.npy')
    #
    # f_data2 = NpyAppendArray(f'./index/{pq_num}/data_2.npy')
    # f_label2 = NpyAppendArray(f'./index/{pq_num}/label_2.npy')
    #
    # f_data3 = NpyAppendArray(f'./index/{pq_num}/data_3.npy')
    # f_label3 = NpyAppendArray(f'./index/{pq_num}/label_3.npy')
    #
    # f_data4 = NpyAppendArray(f'./index/{pq_num}/data_4.npy')
    # f_label4 = NpyAppendArray(f'./index/{pq_num}/label_4.npy')
    #
    # f_data5 = NpyAppendArray(f'./index/{pq_num}/data_5.npy')
    # f_label5 = NpyAppendArray(f'./index/{pq_num}/label_5.npy')
    #
    # f_data6 = NpyAppendArray(f'./index/{pq_num}/data_6.npy')
    # f_label6 = NpyAppendArray(f'./index/{pq_num}/label_6.npy')
    #
    # f_data7 = NpyAppendArray(f'./index/{pq_num}/data_7.npy')
    # f_label7 = NpyAppendArray(f'./index/{pq_num}/label_7.npy')
    #
    # f_data8 = NpyAppendArray(f'./index/{pq_num}/data_8.npy')
    # f_label8 = NpyAppendArray(f'./index/{pq_num}/label_8.npy')
    #
    # f_data9 = NpyAppendArray(f'./index/{pq_num}/data_9.npy')
    # f_label9 = NpyAppendArray(f'./index/{pq_num}/label_9.npy')
    #
    # f_data10 = NpyAppendArray(f'./index/{pq_num}/data_10.npy')
    # f_label10 = NpyAppendArray(f'./index/{pq_num}/label_10.npy')

    for it, data in enumerate(dataloader):
        # -------------------------------- data -------------------------------- #
        img = data["img"].to(device, non_blocking=True)
        aug_img = data["aug_img"].to(device, non_blocking=True)
        label = data["label"].to(device, non_blocking=True)
        img_path = data["img_path"]
        # -------------------------------- loss -------------------------------- #
        with torch.cuda.amp.autocast(enabled=True):
            _, output, linear_preds = model(img=img, aug_img=aug_img, label=label, is_crf=is_crf)

        linear_m.update(linear_preds.to(device), label)

        #############
        # import torch.nn.functional as F
        # z_quantized_index = z_quantized_index.permute(1, 0, 2, 3).contiguous()
        # z_quantized_index = F.interpolate(z_quantized_index.float(), size=label.shape[-2:], mode="nearest")
        # b, c, h, w = z_quantized_index.shape  # (8, 64, 320, 320)
        # z_quantized_index = z_quantized_index.view(b, c, -1).permute(0, 2, 1)  # (8, 320*320, 64)

        # label_ = label.view(b, -1).contiguous()  # (8, 320*320)
        # if it % 40 == 0:
        #     print(it)
        # if it < 40:
        #     f_data.append(z_quantized_index.cpu().numpy())
        #     f_label.append(label_.cpu().numpy())
        # elif it < 80:
        #     f_data2.append(z_quantized_index.cpu().numpy())
        #     f_label2.append(label_.cpu().numpy())
        # elif it < 120:
        #     f_data3.append(z_quantized_index.cpu().numpy())
        #     f_label3.append(label_.cpu().numpy())
        # elif it < 160:
        #     f_data4.append(z_quantized_index.cpu().numpy())
        #     f_label4.append(label_.cpu().numpy())
        # elif it < 200:
        #     f_data5.append(z_quantized_index.cpu().numpy())
        #     f_label5.append(label_.cpu().numpy())
        # elif it < 240:
        #     f_data6.append(z_quantized_index.cpu().numpy())
        #     f_label6.append(label_.cpu().numpy())
        # elif it < 280:
        #     f_data7.append(z_quantized_index.cpu().numpy())
        #     f_label7.append(label_.cpu().numpy())
        # elif it < 320:
        #     f_data8.append(z_quantized_index.cpu().numpy())
        #     f_label8.append(label_.cpu().numpy())
        # elif it < 360:
        #     f_data9.append(z_quantized_index.cpu().numpy())
        #     f_label9.append(label_.cpu().numpy())
        # else:
        #     f_data10.append(z_quantized_index.cpu().numpy())
        #     f_label10.append(label_.cpu().numpy())

        for k, v in output.items():
            if k not in result:
                result[k] = v
            else:
                result[k] += v
        count += 1

        if cfg["is_visualize"] and is_crf:
            os.makedirs(cfg["visualize_path"], exist_ok=True)
            saved_data["img_path"].append("".join(img_path))
            saved_data["linear_preds"].append(linear_preds.cpu().squeeze(0))
            saved_data["label"].append(label.cpu().squeeze(0))

            #     dataset_name = cfg["dataset_name"]
            #     pq_visualization(save_dir=f"./visualize/pq/{dataset_name}/vq/",
            #                      saved_data=z_quantized_index,
            #                      img_path=img_path
            #                      )

    barrier()
    linear_result = linear_m.compute("linear")  # {iou, accuracy}

    if cfg["is_visualize"] and is_crf:
        visualization(cfg["visualize_path"] + "/" + str(current_iter), cfg["dataset_name"], saved_data)

    barrier()
    for k, v in result.items():
        result[k] /= count
    result = all_reduce_dict(result, op="mean")

    valid_time = time.time() - valid_start_time

    if is_master():
        s += f"[Linear] mIoU {linear_result['iou'].item():.6f}, acc: {linear_result['accuracy'].item():.6f}\n"
        for k, v in result.items():
            s += f"... {k}: {v.item() if isinstance(v, torch.Tensor) else v:.6f}\n"
        s += f"... time: {valid_time:.3f} sec"

        print(s)
        log_dict = {
            "iters": current_iter,
            "Linear_mIoU": linear_result['iou'].item(),
            "Linear_Accuracy": linear_result['accuracy'].item(),
        }
        for k, v in result.items():
            k = "VAL_" + k
            log_dict[k] = v.item() if isinstance(v, torch.Tensor) else v

        if not is_crf:
            wandb.log(log_dict)

    return result, linear_result


def run(cfg: Dict, debug: bool = False) -> None:
    # ======================================================================================== #
    # Initialize
    # ======================================================================================== #
    scaler = torch.cuda.amp.GradScaler(init_scale=2048, growth_interval=1000, enabled=True)

    device, local_rank = set_dist(device_type="cuda")
    if is_master():
        pprint.pprint(cfg)  # print config to check if all arguments are correctly given.
    save_dir = set_wandb(cfg, force_mode="disabled" if debug else None)
    set_seed(seed=cfg["seed"])
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
    model = build_model(cfg, name=cfg["wandb"]["name"].lower(), world_size=get_world_size())
    model = model.to(device)
    if is_distributed_set():
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[local_rank],
                                        output_device=device)
        model_m = model.module  # actual model without wrapping

    else:
        model_m = model

    if is_master():
        print(model)
        p1, p2 = count_params(model_m.model.parameters(), requires_grad=True)
        print(f"Model parameters: {p1} tensors, {p2} elements.")

    if cfg["resume"]["checkpoint"] is not None:
        save_dir = cfg["resume"]["checkpoint"]
        ckpt = True
    else:
        ckpt = None

    # ======================================================================================== #
    # Optimizer & Scheduler
    # ======================================================================================== #
    model_params = split_params_for_optimizer(model_m.model, cfg["optimizer"])

    model_optimizer = build_optimizer(cfg["optimizer"]["model"], model_params)

    optimizers = [model_optimizer]

    iter_per_epoch = len(train_dataloader)
    num_accum = cfg["train"].get("num_accum", 1)
    model_scheduler = build_scheduler(cfg["scheduler"]["model"], model_optimizer, iter_per_epoch, num_accum,
                                      epoch=cfg["train"]["max_epochs"])

    schedulers = [model_scheduler]

    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.momentum /= num_accum

    # ======================================================================================== #
    # Trainer
    # ======================================================================================== #

    # -------- config -------- #
    train_cfg = cfg["train"]
    max_epochs = train_cfg["max_epochs"]

    # -------- status -------- #
    best_metric = dict()
    best_metric["Linear_mIoU"] = 0.0
    best_metric["Linear_Accuracy"] = 0.0

    current_epoch = 0
    current_iter = 0
    best_iter = best_epoch = 0

    if ckpt is not None:
        current_epoch = max_epochs

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
        current_iter, best_metric, best_epoch, best_iter = train_epoch(model, optimizers,
                                                                       schedulers,
                                                                       train_dataloader,
                                                                       valid_dataloader,
                                                                       cfg, device, save_dir, current_epoch,
                                                                       current_iter, best_metric, best_epoch, best_iter,
                                                                       scaler)
        epoch_time = time.time() - epoch_start_time

        if is_master():
            s = time_log()
            s += f"End train epoch {current_epoch} / {max_epochs}\n"
            s += f"... time: {epoch_time:.3f} sec"
            print(s)

        barrier()
        current_epoch += 1

    # -------- final evaluation -------- #
    print("final evaluation")
    s = time_log()
    best_checkpoint = torch.load(f"{save_dir}/best.pth", map_location=device)
    model_m = model.module if isinstance(model, DistributedDataParallel) else model
    model_m.load_state_dict(best_checkpoint['model'], strict=True)
    final_start_time = time.time()
    s += "Final evaluation (before CRF)\n"
    _, linear_result = valid_epoch(model_m, valid_dataloader, cfg, device, current_iter, is_crf=False)
    s += f"[Linear] mIoU {linear_result['iou'].item():.6f}, acc: {linear_result['accuracy'].item():.6f}\n"
    s += time_log()
    s += "Final evaluation (after CRF)\n"
    _, linear_result = valid_epoch(model_m, valid_dataloader, cfg, device, current_iter, is_crf=True)
    s += f"[Linear] mIoU {linear_result['iou'].item():.6f}, acc: {linear_result['accuracy'].item():.6f}\n"

    final_time = time.time() - final_start_time
    s += f"... time: {final_time:.3f} sec"

    if is_master():
        print(s)
        wandb.finish()
    barrier()


if __name__ == '__main__':
    args, config = prepare_config()
    run(config, args.debug)
