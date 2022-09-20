from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from model.dino.dino_featurizer import DinoFeaturizer
from model.blocks.resnet import EncResBlock, DecResBlock, LayerNorm2d


class DINORes(nn.Module):
    def __init__(self, cfg: dict):  # cfg["model"]
        super().__init__()
        self.cfg = cfg

        self.extractor = DinoFeaturizer(cfg["pretrained"])
        self.feat_dim = self.extractor.n_feats  # 384
        self.hidden_dim = cfg.get("hidden_dim", self.feat_dim)

        # -------- semantic-encoder -------- #
        num_enc_blocks = cfg["enc_num_blocks"]
        semantic_enc_proj = []
        for i in range(num_enc_blocks):
            semantic_enc_proj.append(EncResBlock(self.feat_dim if (i == 0) else self.hidden_dim, self.hidden_dim))
        self.semantic_enc_proj = nn.Sequential(*semantic_enc_proj)

        # -------- local-encoder -------- #
        local_enc_proj = []
        for i in range(num_enc_blocks):
            local_enc_proj.append(EncResBlock(self.feat_dim if (i == 0) else self.hidden_dim, self.hidden_dim))
        self.local_enc_proj = nn.Sequential(*local_enc_proj)

        # -------- decoder -------- #
        num_dec_blocks = cfg["dec_num_blocks"]
        dec_proj = []
        for i in range(num_dec_blocks):
            dec_proj.append(
                DecResBlock(self.hidden_dim, self.feat_dim if (i == num_dec_blocks - 1) else self.hidden_dim))
        self.dec_proj = nn.Sequential(*dec_proj)

        last_norm = cfg.get("last_norm", False)
        self.dec_norm = LayerNorm2d(self.feat_dim) if last_norm else None

        self.agg_type = cfg["agg_type"]
        if (self.agg_type == "cat") or (self.agg_type == "concat"):
            self.agg_type = "concat"
            self.aggregate_proj = nn.Conv2d(self.hidden_dim + self.hidden_dim, self.hidden_dim, 1, 1, 0)
        elif (self.agg_type == "add") or (self.agg_type == "sum"):
            self.agg_type = "add"
            self.aggregate_proj = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1, 1, 0)
        else:
            raise ValueError(f"Unsupported aggregate type {self.agg_type}.")

    def forward(self, img: torch.Tensor
                ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        dino_feat = self.extractor(img)  # (b, 384, 28, 28) (b, d, h, w)
        output = dict()
        with torch.autograd.set_detect_anomaly(True):
            semantic_feat = self.semantic_enc_proj(dino_feat)  # (b, hidden_d, h, w)
            local_feat = self.local_enc_proj(dino_feat)  # (b, hidden_d, h, w)

            if self.agg_type == "concat":
                feat = torch.cat([semantic_feat, local_feat], dim=1)
            elif self.agg_type == "add":
                feat = semantic_feat + local_feat
            else:
                raise ValueError

            feat = self.aggregate_proj(feat)  # (b, 384, 28, 28)
            recon = self.dec_proj(feat)  # (b, 384, 28, 28)

            if self.dec_norm is not None:
                recon = self.dec_norm(recon)

            recon_loss = F.mse_loss(recon, dino_feat)

            output["recon-loss"] = recon_loss

            return dino_feat, semantic_feat, output
