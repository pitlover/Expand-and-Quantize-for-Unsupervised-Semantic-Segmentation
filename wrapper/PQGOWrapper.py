from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.dino_unseg import DINOUnSeg
from model.evaluator import UnSegEvaluator

__all__ = [
    "PQGOWrapper"
]


class PQGOWrapper(nn.Module):

    def __init__(self,
                 cfg,
                 model: DINOUnSeg,
                 ) -> None:
        super().__init__()
        # cfg = cfg

        self.model = model

        self.num_classes = cfg["num_classes"]
        self.extra_classes = cfg["eval"]["extra_classes"]

        # -------- Loss weight --------- #
        self.stego_weight = cfg["loss"]["stego_weight"]
        self.recon_weight = cfg["loss"].get("recon_weight", 0.0)
        self.cls_weight = cfg["loss"].get("cls_weight", 0.0)
        self.mse_weight = cfg["loss"].get("mse_weight", 0.0)
        self.vq_weight = cfg["loss"]["vq_weight"]


        self.output_type = cfg["eval"]["output_type"]
        self.use_kmeans_sampling = cfg["model"]["vq"]["use_kmeans_sampling"]

        if self.output_type == "feat":
            output_dim = cfg["model"]["vq"]["embed_dims"][0]
            # output_dim = self.model.feat_dim
        elif "vq" == self.output_type[:2]:
            vq_idx = int(self.output_type[2:])
            output_dim = cfg["model"]["vq"]["embed_dims"][vq_idx]
        else:
            raise ValueError(f"Unsupported output type {self.output_type}.")

        self.output_dim = output_dim

        self.evaluator = UnSegEvaluator(
            output_dim, self.num_classes, self.extra_classes
        )

    def forward(self,
                img: torch.Tensor,
                aug_img: torch.Tensor,
                label: torch.Tensor,
                img_pos: torch.Tensor = None,
                it: int = -1,
                is_crf: bool = False,
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        model_loss = torch.zeros(1, device=img.device)
        b, _, H, W = img.shape
        output = dict()

        code, feat_vqs, output = self.model(img=img, aug_img=aug_img, img_pos=img_pos, it=it)

        # feat: (b, 384, 28, 28)
        # vqs: (b, vq_k0, 28, 28), (b, vq_k1, 28, 28), ...
        # output: {vq0-current-p10/50/90 , vq0-total-p10/50/90, vq0-loss, vq0-~loss, ..., recon-loss}
        if self.training and self.stego_weight > 0.0:
            model_loss = model_loss + (output["stego-loss"] * self.stego_weight)

        if self.recon_weight > 0.0:
            model_loss = model_loss + output["recon-loss"] * self.recon_weight

        if self.vq_weight > 0.0:
            model_loss = model_loss + (output["vq-loss"] * self.vq_weight)

        if self.cls_weight > 0.0:
            model_loss = model_loss + (output["cls-loss"] * self.cls_weight)

        if self.mse_weight > 0.0:
            model_loss = model_loss + (output["mse-loss"] * self.mse_weight)

        output["loss"] = model_loss

        if self.output_type == "feat":
            out = torch.clone(code).detach()
        elif "vq" == self.output_type[:2]:
            out = torch.clone(feat_vqs).detach()  # (b, d, h, w)
        else:
            raise ValueError(f"Unsupported output type {self.output_type}.")


        linear_loss, linear_preds, cluster_loss, cluster_preds = self.evaluator(
            out, img, label=label, is_crf=is_crf)
        output["linear-loss"] = linear_loss
        output["cluster-loss"] = cluster_loss

        total_loss = model_loss + linear_loss + cluster_loss

        return total_loss, output, (linear_preds, cluster_preds)
