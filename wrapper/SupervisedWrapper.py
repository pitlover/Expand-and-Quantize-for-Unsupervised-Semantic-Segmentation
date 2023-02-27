from typing import Dict, Tuple
import torch
import torch.nn as nn
from model.dino_stego import DINOStego
from utils.crf_utils import batched_crf
import torch.nn.functional as F

__all__ = [
    "SupervisedWrapper"
]


class SupervisedWrapper(nn.Module):

    def __init__(self,
                 cfg,
                 model: DINOStego,
                 ) -> None:
        super().__init__()
        # cfg = cfg

        self.model = model

        self.num_classes = cfg["num_classes"]
        self.extra_classes = cfg["eval"]["extra_classes"]

        self.output_dim = cfg["model"]["pretrained"]["dim"]

        self.linear_probe = LinearProbe(self.output_dim, self.num_classes)

    def forward(self,
                img: torch.Tensor,
                aug_img: torch.Tensor,
                label: torch.Tensor,
                img_pos: torch.Tensor = None,
                img_path: str = None,
                it: int = -1,
                is_crf: bool = False,
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        dino_feat, code, output, _ = self.model(img, img_pos)

        linear_loss, linear_preds = self.linear_probe(code, img, label=label, is_crf=is_crf)
        output["ce-loss"] = linear_loss

        total_loss = linear_loss

        return total_loss, output, linear_preds


class LinearProbe(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_classes: int
                 ):
        super().__init__()

        self.num_classes = num_classes
        self.linear_probe = nn.Conv2d(embed_dim, num_classes, kernel_size=1, stride=1)
        self.linear_loss = nn.CrossEntropyLoss()

    def forward(self, out: torch.Tensor, img, label: torch.Tensor, is_crf: bool):

        if out.shape[-2:] != label.shape[-2:]:
            out = F.interpolate(out, label.shape[-2:], mode="bilinear", align_corners=False)

        if is_crf:
            linear_log_prob = torch.log_softmax(self.linear_probe(out), dim=1)
            linear_preds = batched_crf(img, linear_log_prob).argmax(1)
            linear_loss = torch.zeros()

        else:
            linear_logits = self.linear_probe(out)
            linear_preds = linear_logits.argmax(1)

            label_flat = label.reshape(-1)
            mask = torch.logical_and(label_flat >= 0, label_flat < self.num_classes)

            linear_logits = F.interpolate(linear_logits, label.shape[-2:], mode="bilinear", align_corners=False)
            logit_flat = linear_logits.permute(0, 2, 3, 1).reshape(-1, self.num_classes)

            label_flat = label_flat[mask]
            logit_flat = logit_flat[mask]

            linear_loss = self.linear_loss(logit_flat, label_flat).mean()

        return linear_loss, linear_preds
