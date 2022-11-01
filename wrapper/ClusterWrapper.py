from typing import Dict, Tuple
import torch
import torch.nn as nn
from model.dino_stego import DINOStego
from model.evaluator import UnSegEvaluator
import torch.nn.functional as F

__all__ = [
    "ClusterWrapper"
]


class ClusterWrapper(nn.Module):

    def __init__(self,
                 cfg,
                 model: DINOStego,
                 ) -> None:
        super().__init__()
        # cfg = cfg

        self.model = model

        self.num_classes = cfg["num_classes"]
        self.extra_classes = cfg["eval"]["extra_classes"]

        self.contra_pos_weight = cfg["loss"].get("info_nce_weight", 0.0)
        self.cluster_weight = cfg["loss"].get("swav_weight", 0.0)
        self.margin_weight = cfg["loss"]["margin_weight"]

        self.output_dim = cfg["model"]["hidden_dim"]

        self.evaluator = UnSegEvaluator(
            self.output_dim, self.num_classes, self.extra_classes
        )

    def forward(self,
                img: torch.Tensor,
                aug_img : torch.Tensor,
                label: torch.Tensor,
                queue: torch.Tensor = None,
                is_crf: bool = False,
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:

        dino_feat, semantic_feat, out_prototypes, output = self.model(img, aug_img, queue)

        model_loss = torch.zeros(1, device=img.device)

        if self.training:
            if self.contra_pos_weight > 0.0:
                model_loss += (output["contra-loss-pos"] * self.contra_pos_weight)
            if self.cluster_weight > 0.0:
                model_loss += (output["swav-loss"] * self.cluster_weight)
            if self.margin_weight > 0.0:
                model_loss = model_loss + (output["margin"] * self.margin_weight)

        output["loss"] = model_loss

        out = semantic_feat.detach()

        linear_loss, linear_preds, cluster_loss, cluster_preds = self.evaluator(
            out, img, label=label, is_crf=is_crf)
        output["linear-loss"] = linear_loss
        output["cluster-loss"] = cluster_loss

        total_loss = model_loss + linear_loss + cluster_loss

        return total_loss, output, (linear_preds, cluster_preds)
