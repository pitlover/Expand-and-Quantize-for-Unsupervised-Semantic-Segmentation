from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from model.dino.dino_featurizer import DinoFeaturizer
from model.loss import STEGOLoss


class DINOStego(nn.Module):
    def __init__(self, cfg: dict):  # cfg["model"]
        super().__init__()
        self.cfg = cfg

        self.extractor = DinoFeaturizer(cfg["pretrained"])
        self.feat_dim = self.extractor.n_feats  # 384
        self.dim = cfg["pretrained"]["dim"]
        self.dropout = torch.nn.Dropout2d(p=.1)

        # -------- head -------- #
        self.cluster1 = self.make_clusterer(self.feat_dim)
        self.cluster2 = self.make_nonlinear_clusterer(self.feat_dim)

        self.corr_loss = STEGOLoss(cfg)

    def make_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))

    def make_nonlinear_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))

    def forward(self, img: torch.Tensor, pos_img: torch.Tensor
                ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        output = dict()

        dino_feat = self.extractor(img)  # (b, 384, 28, 28) (b, d, h, w)
        code = self.cluster1(self.dropout(dino_feat))
        code += self.cluster2(self.dropout(dino_feat))

        dino_feat_pos = self.extractor(pos_img)  # (b, 384, 28, 28) (b, d, h, w)
        code_pos = self.cluster1(self.dropout(dino_feat_pos))
        code_pos += self.cluster2(self.dropout(dino_feat_pos))

        output["stego-loss"], output["self-loss"], \
        output["knn-loss"], output["rand-loss"] = self.corr_loss(dino_feat,
                                                                 dino_feat_pos,
                                                                 code,
                                                                 code_pos)
        return dino_feat, code, output
