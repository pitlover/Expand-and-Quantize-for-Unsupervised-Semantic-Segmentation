from typing import Dict, Tuple
import torch
import torch.nn as nn

from model.dino_unseg import DINOUnSeg
from model.evaluator import UnSegEvaluator

__all__ = [
    "DINOUnSegWrapper"
]


class DINOUnSegWrapper(nn.Module):

    def __init__(self,
                 cfg,
                 model: DINOUnSeg,
                 ) -> None:
        super().__init__()
        # cfg = cfg

        self.model = model

        self.num_classes = cfg["num_classes"]
        self.extra_classes = cfg["eval"]["extra_classes"]

        self.num_vq = self.model.num_vq
        self.recon_weight = cfg["loss"]["recon_weight"]
        self.vq_weight = cfg["loss"]["vq_weight"]
        self.contra_weight = cfg["loss"].get("contra_weight", 0.0)
        self.stego_weight = cfg["loss"].get("stego_weight", 0.0)
        self.output_type = cfg["eval"]["output_type"]

        if self.output_type == "feat":
            output_dim = self.model.feat_dim
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
                label: torch.Tensor,
                is_crf: bool = False,
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        feat, feat_vqs, output = self.model(img)
        # feat: (b, 384, 28, 28)
        # vqs: (b, vq_k0, 28, 28), (b, vq_k1, 28, 28), ...
        # output: {vq0-current-p10/50/90 , vq0-total-p10/50/90, vq0-loss, vq0-~loss, ..., recon-loss}

        model_loss = output["recon-loss"] * self.recon_weight

        # if self.contra_weight > 0:
        #     model_loss += (output["contra-loss"] * self.contra_weight)
        model_loss += (output["contra-loss"] * self.contra_weight)

        if self.stego_weight > 0:
            model_loss += (output["stego-loss"] * self.stego_weight)

        for i in range(self.num_vq):
            model_loss = model_loss + (output[f"vq{i}-loss"] * self.vq_weight)

        output["loss"] = model_loss

        if self.output_type == "feat":
            out = feat.detach()
        elif "vq" == self.output_type[:2]:
            vq_idx = int(self.output_type[2:])
            out = feat_vqs[vq_idx].detach()
        else:
            raise ValueError(f"Unsupported output type {self.output_type}.")

        linear_loss, linear_preds, cluster_loss, cluster_preds = self.evaluator(
            out, img, label=label, is_crf=is_crf)
        output["linear-loss"] = linear_loss
        output["cluster-loss"] = cluster_loss

        total_loss = model_loss + linear_loss + cluster_loss

        return total_loss, output, (linear_preds, cluster_preds)

    def restart(self):
        self.model.restart()
