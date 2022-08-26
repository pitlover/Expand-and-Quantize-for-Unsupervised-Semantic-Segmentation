from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from model.dino.dino_featurizer import DinoFeaturizer
from loss import ContrastiveCorrelationLoss

class DINOStego(nn.Module):
    def __init__(self, cfg: dict):  # cfg["model"]
        super().__init__()
        self.cfg = cfg

        self.extractor = DinoFeaturizer(cfg["pretrained"])
        self.feat_dim = self.extractor.n_feats  # 384

        # -------- head -------- #
        self.cluster1 = self.make_clusterer(self.n_feats)
        self.cluster2 = self.make_nonlinear_clusterer(self.n_feats)

    def make_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))

    def make_nonlinear_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))

    def forward(self, img: torch.Tensor
                ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        dino_feat = self.extractor(img)  # (b, 384, 28, 28) (b, d, h, w)

        code = self.cluster1(self.dropout(dino_feat))

        if self.proj_type == "nonlinear":
            code += self.cluster2(self.dropout(dino_feat))

        output = dict()
        output["stego-loss"] = ContrastiveCorrelationLoss()(dino_feat, dino_feat_pos, code, code_pos)

        return dino_feat, code, output

    def restart(self):
        for i in range(self.num_vq):
            self.vq_blocks[i].restart()
