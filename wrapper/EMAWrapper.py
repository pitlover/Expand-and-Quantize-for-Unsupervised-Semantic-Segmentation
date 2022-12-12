from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.dino_unseg import DINOUnSeg
from model.evaluator import UnSegEvaluator

__all__ = [
    "EMAWrapper"
]


class EMAWrapper(nn.Module):

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
        self.mse_weight = cfg["loss"]["mse_weight"]
        self.bank_weight = cfg["loss"]["info_nce_weight"]
        self.output_type = cfg["eval"]["output_type"]

        if self.output_type == "feat":
            output_dim = cfg["model"]["hidden_dim"]

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
        code, other_code, output = self.model(img=img, aug_img=aug_img, img_pos=img_pos, it=it)

        if self.training and self.stego_weight > 0.0:
            model_loss = model_loss + (output["stego-loss"] * self.stego_weight)

        if self.mse_weight > 0.0:
            model_loss = model_loss + (output["mse-loss"] * self.mse_weight)

        if self.bank_weight > 0.0:
            model_loss = model_loss + (output["info-nce"] * self.bank_weight)

        output["loss"] = model_loss

        if self.output_type == "feat":
            out = torch.clone(code).detach()
        else:
            raise ValueError(f"Unsupported output type {self.output_type}.")

        linear_loss, linear_preds, cluster_loss, cluster_preds = self.evaluator(
            out, img, label=label, is_crf=is_crf)

        output["linear-loss"] = linear_loss
        output["cluster-loss"] = cluster_loss
        total_loss = model_loss + linear_loss + cluster_loss

        return total_loss, output, (linear_preds, cluster_preds)
