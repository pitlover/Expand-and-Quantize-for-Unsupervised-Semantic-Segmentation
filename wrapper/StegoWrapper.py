from typing import Dict, Tuple
import torch
import torch.nn as nn
from model.dino_stego import DINOStego
from model.evaluator import UnSegEvaluator

__all__ = [
    "StegoWrapper"
]


class StegoWrapper(nn.Module):

    def __init__(self,
                 cfg,
                 model: DINOStego,
                 ) -> None:
        super().__init__()
        # cfg = cfg

        self.model = model

        self.num_classes = cfg["num_classes"]
        self.extra_classes = cfg["eval"]["extra_classes"]

        self.stego_weight = cfg["loss"].get("stego_weight", 1.0)

        output_dim = cfg["model"]["pretrained"]["dim"]

        self.output_dim = output_dim
        self.evaluator = UnSegEvaluator(
            output_dim, self.num_classes, self.extra_classes
        )

    def forward(self,
                img: torch.Tensor,
                label: torch.Tensor,
                img_pos: torch.Tensor,
                is_crf: bool = False,
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        dino_feat, code, output = self.model(img, img_pos)
        model_loss = output["stego-loss"] * self.stego_weight
        output["loss"] = model_loss

        out = code.detach()

        linear_loss, linear_preds, cluster_loss, cluster_preds = self.evaluator(
            out, img, label=label, is_crf=is_crf)
        output["linear-loss"] = linear_loss
        output["cluster-loss"] = cluster_loss

        total_loss = model_loss + linear_loss + cluster_loss

        return total_loss, output, (linear_preds, cluster_preds)
