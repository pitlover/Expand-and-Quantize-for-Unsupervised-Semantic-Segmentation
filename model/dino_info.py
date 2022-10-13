from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from model.dino.dino_featurizer import DinoFeaturizer
from model.blocks.resnet_linear import EncResBlock, DecResBlock
from model.quantizer import VectorQuantizer, EMAVectorQuantizer


class DINOInfo(nn.Module):
    def __init__(self, cfg: dict):  # cfg["model"]
        super().__init__()
        self.cfg = cfg

        self.extractor = DinoFeaturizer(cfg["pretrained"])
        self.feat_dim = self.extractor.n_feats  # 384

        # -------- encoder -------- #
        num_enc_blocks = cfg["enc_num_blocks"]
        enc_proj = []
        for i in range(num_enc_blocks):
            enc_proj.append(EncResBlock(self.feat_dim, self.feat_dim))
        self.enc_proj = nn.Sequential(*enc_proj)

        # -------- vq -------- #
        vq_num_codebooks = cfg["vq"]["num_codebooks"]
        vq_embed_dims = cfg["vq"]["embed_dims"]
        assert len(vq_num_codebooks) == len(vq_embed_dims)
        self.num_vq = len(vq_num_codebooks)
        self.beta = cfg["vq"]["beta"]
        self.normalize = cfg["vq"]["normalize"]
        self.vq_type = cfg["vq"]["vq_type"]
        self.use_restart = cfg["vq"].get("use_restart", False)
        self.use_gumbel = cfg["vq"].get("use_gumbel", False)

        vq_kwargs = dict(beta=self.beta, normalize=self.normalize,
                         use_restart=self.use_restart, use_gumbel=self.use_gumbel)

        if self.vq_type == "ema":
            vq_kwargs["decay"] = cfg["vq"]["decay"]
            vq_kwargs["eps"] = cfg["vq"]["eps"]
            vq_blocks = [
                EMAVectorQuantizer(vq_num_codebooks[i], vq_embed_dims[i], **vq_kwargs)
                for i in range(self.num_vq)
            ]
        elif self.vq_type == "param":
            vq_blocks = [
                VectorQuantizer(vq_num_codebooks[i], vq_embed_dims[i], **vq_kwargs)
                for i in range(self.num_vq)
            ]
        else:
            raise ValueError(f"Unsupported vq type {self.vq_type}.")
        self.vq_blocks = nn.ModuleList(vq_blocks)

        # -------- vq connections -------- #
        vq_input_proj = []
        for i in range(self.num_vq):
            vq_input_proj.append(nn.Sequential(
                nn.Conv2d(self.feat_dim, vq_embed_dims[i], 1, 1, 0, bias=False),
                nn.BatchNorm2d(vq_embed_dims[i])
            ))
        self.vq_input_proj = nn.ModuleList(vq_input_proj)

        vq_output_proj = []
        for i in range(self.num_vq):
            vq_output_proj.append(nn.Sequential(
                nn.Conv2d(self.feat_dim, self.feat_dim, 1, 1, 0),
                nn.ReLU(inplace=True),
                # nn.Conv2d(self.feat_dim, self.feat_dim, 1, 1, 0),
                # nn.ReLU(inplace=True)
            ))
        self.vq_output_proj = nn.ModuleList(vq_output_proj)

        self.vq_concat_proj = nn.Conv2d(sum(vq_embed_dims) + self.feat_dim, self.feat_dim, 1, 1, 0)

        # -------- decoder -------- #
        num_dec_blocks = cfg["dec_num_blocks"]
        dec_proj = []
        for i in range(num_dec_blocks):
            dec_proj.append(DecResBlock(self.feat_dim, self.feat_dim))
        self.dec_proj = nn.Sequential(*dec_proj)
        self.dec_norm = nn.LayerNorm(self.feat_dim)

    def forward(self, img: torch.Tensor
                ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        dino_feat = self.extractor(img)  # (b, 384, 28, 28) (b, d, h, w)

        feat = self.enc_proj(dino_feat)  # (b, 384, 28, 28)

        output = dict()
        feat_vqs = []

        for i in range(self.num_vq):
            feat_i = self.vq_input_proj[i](feat)
            feat_vq_i, vq_i_output, distance_prob = self.vq_blocks[i](feat_i)
            feat_vqs.append(feat_vq_i)

            for k, v in vq_i_output.items():
                output[f"vq{i}-{k}"] = v

            feat = self.vq_output_proj[i](feat)

        # last
        feat_vqs.append(feat)

        feat = torch.cat(feat_vqs, dim=1)
        feat = self.vq_concat_proj(feat)  # (b, 384, 28, 28)

        recon = self.dec_proj(feat)  # (b, 384, 28, 28)

        recon = recon.permute(0, 2, 3, 1).contiguous()
        recon = self.dec_norm(recon)
        recon = recon.permute(0, 3, 1, 2).contiguous()

        recon_loss = F.mse_loss(recon, dino_feat)

        output["recon-loss"] = recon_loss

        return feat, feat_vqs, output

    def restart(self):
        for i in range(self.num_vq):
            self.vq_blocks[i].restart()
