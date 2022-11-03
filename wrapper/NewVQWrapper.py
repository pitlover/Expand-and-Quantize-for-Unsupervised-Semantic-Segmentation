from typing import Dict, Tuple
import torch
import torch.nn as nn

from model.dino_unseg import DINOUnSeg
from model.evaluator import UnSegEvaluator

__all__ = [
    "DINONewVQWrapper"
]


class DINONewVQWrapper(nn.Module):

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
        # -------- Loss weight --------- #
        self.recon_weight = cfg["loss"]["recon_weight"]
        self.vq_weight = cfg["loss"]["vq_weight"]
        self.info_nce_weight = cfg["loss"]["info_nce_weight"]
        self.jsd_weight = cfg["loss"]["jsd_weight"]
        self.margin_weight = cfg["loss"]["margin_weight"]
        self.entropy_weight = 0.0
        if self.jsd_weight > 0.0:
            self.entropy_weight = cfg["loss"]["jsd"]["entropy_weight"]

        self.output_type = cfg["eval"]["output_type"]
        self.use_kmeans_sampling = cfg["model"]["vq"]["use_kmeans_sampling"]

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
                aug_img: torch.Tensor,
                label: torch.Tensor,
                it: int,
                is_crf: bool = False,
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        model_loss = torch.zeros(1, device=img.device)
        b, _, H, W = img.shape
        output = dict()
        if not self.use_kmeans_sampling:
            feat, feat_vqs, output = self.model(img, aug_img, it)
            # feat: (b, 384, 28, 28)
            # vqs: (b, vq_k0, 28, 28), (b, vq_k1, 28, 28), ...
            # output: {vq0-current-p10/50/90 , vq0-total-p10/50/90, vq0-loss, vq0-~loss, ..., recon-loss}
            model_loss = 0
            if self.recon_weight > 0.0:
                model_loss = output["recon-loss"] * self.recon_weight

            if self.vq_weight > 0.0:
                model_loss = model_loss + (output["vq-loss"] * self.vq_weight)

            if self.info_nce_weight > 0.0:
                model_loss = model_loss + (output["info_nce"] * self.info_nce_weight)

            if self.jsd_weight > 0.0:
                model_loss = model_loss + (output["jsd"] * self.jsd_weight)

                if self.entropy_weight > 0.0:
                    model_loss += (output["entropy"] * self.entropy_weight)
            if self.margin_weight > 0.0:
                model_loss = model_loss + (output["margin"] * self.margin_weight)

            output["loss"] = model_loss

        else:  # k-means sampling
            if self.training:
                _, _, output = self.model(img, stage=1)
                model_loss = output["recon-loss"] * self.recon_weight
                model_loss = model_loss + (output["vq-loss"] * self.vq_weight)

                if self.info_nce_weight > 0.0:
                    model_loss += (output["info_nce"] * self.info_nce_weight)

                    if self.entropy_weight > 0.0:
                        model_loss += (output["entropy"] * self.entropy_weight)
                output["loss"] = model_loss

            with torch.no_grad():
                feat, feat_vqs, _ = self.model(img)

        if self.output_type == "feat":
            out = feat.detach()
        elif "vq" == self.output_type[:2]:
            out = feat_vqs.detach()  # (b, d, h, w)
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
