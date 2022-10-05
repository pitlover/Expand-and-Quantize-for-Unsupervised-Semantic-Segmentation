from typing import Dict, Tuple
import torch
import torch.nn as nn
from model.dino_stego import DINOStego
from model.evaluator import UnSegEvaluator

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

        self.output_dim = cfg["model"]["hidden_dim"]

        self.evaluator = UnSegEvaluator(
            self.output_dim, self.num_classes, self.extra_classes
        )

    def forward(self,
                img: torch.Tensor,
                label: torch.Tensor,
                club_optimizer=None,
                is_crf: bool = False,
                scaler=None
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        dino_feat, semantic_feat, output = self.model(img, club_optimizer, scaler)

        model_loss = 0

        if self.training:
            if self.contra_pos_weight > 0.0:
                model_loss = model_loss + (output["contra-loss-pos"] * self.contra_pos_weight)

        output["loss"] = model_loss

        out = semantic_feat.detach()

        linear_loss, linear_preds, cluster_loss, cluster_preds = self.evaluator(
            out, img, label=label, is_crf=is_crf)
        output["linear-loss"] = linear_loss
        output["cluster-loss"] = cluster_loss

        total_loss = model_loss + linear_loss + cluster_loss

        return total_loss, output, (linear_preds, cluster_preds)
