import random
from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from model.dino.dino_featurizer import DinoFeaturizer
from model.blocks.resnet import EncResBlock, DecResBlock, LayerNorm2d
from model.loss import JSDLoss
from model.quantizer import VectorQuantizer, EMAVectorQuantizer, ProductQuantizerWrapper

import torchvision.transforms as transforms


class DINOVae(nn.Module):
    def __init__(self, cfg: dict):  # cfg["model"]
        super().__init__()
        self.cfg = cfg

        self.extractor = DinoFeaturizer(cfg["pretrained"])
        self.feat_dim = self.extractor.n_feats  # 384
        self.hidden_dim = cfg.get("hidden_dim", self.feat_dim)

        # -------- encoder -------- #
        num_enc_blocks = cfg["enc_num_blocks"]
        enc_proj = []
        for i in range(num_enc_blocks):
            enc_proj.append(EncResBlock(self.feat_dim if (i == 0) else self.hidden_dim, self.hidden_dim))

        self.enc_proj_bottom = nn.Sequential(*enc_proj)
        self.enc_proj_top = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim // 2, self.hidden_dim, 3, padding=1)
        )

        # -------- vq -------- #
        vq_num_codebooks = cfg["vq"]["num_codebooks"]
        vq_embed_dims = cfg["vq"]["embed_dims"]
        assert len(vq_num_codebooks) == len(vq_embed_dims)
        self.num_vq = len(vq_num_codebooks)
        self.beta = cfg["vq"]["beta"]
        self.normalize = cfg["vq"]["normalize"]
        self.vq_type = cfg["vq"]["vq_type"]
        self.use_restart = cfg["vq"].get("use_restart", False)
        self.use_split = cfg["vq"].get("use_split", False)
        self.use_gumbel = cfg["vq"].get("use_gumbel", False)
        self.use_weighted_sum = cfg["vq"].get("use_weighted_sum", False)
        self.update_norm = cfg["vq"].get("update_norm", True)
        self.jsd = JSDLoss()

        self.num_pq = cfg["vq"].get("num_pq", 1)
        if isinstance(self.num_pq, int):
            self.num_pq = [self.num_pq] * self.num_vq

        vq_kwargs = dict(beta=self.beta, normalize=self.normalize,
                         use_restart=self.use_restart, use_gumbel=self.use_gumbel, use_split=self.use_split,
                         use_weighted_sum=self.use_weighted_sum, update_norm=self.update_norm)

        if self.vq_type == "ema":
            vq_kwargs["decay"] = cfg["vq"]["decay"]
            vq_kwargs["eps"] = cfg["vq"]["eps"]
            vq_blocks = [
                EMAVectorQuantizer(vq_num_codebooks[i], vq_embed_dims[i], **vq_kwargs) if (self.num_pq == 1) else
                ProductQuantizerWrapper(self.num_pq[i], vq_num_codebooks[i], vq_embed_dims[i], **vq_kwargs,
                                        quantizer_cls=EMAVectorQuantizer)
                for i in range(self.num_vq)
            ]
        elif self.vq_type == "param":
            vq_blocks = [
                VectorQuantizer(vq_num_codebooks[i], vq_embed_dims[i], **vq_kwargs) if (self.num_pq == 1) else
                ProductQuantizerWrapper(self.num_pq[i], vq_num_codebooks[i], vq_embed_dims[i], **vq_kwargs,
                                        quantizer_cls=VectorQuantizer)
                for i in range(self.num_vq)
            ]
        else:
            raise ValueError(f"Unsupported vq type {self.vq_type}.")
        self.vq_blocks = nn.ModuleList(vq_blocks)

        # -------- vq connections -------- #
        vq_input_proj = []
        for i in range(self.num_vq):
            vq_input_proj.append(nn.Sequential(
                nn.LeakyReLU(0.1, inplace=False),  # MOVED TO HERE
                nn.Conv2d(self.hidden_dim, vq_embed_dims[i], 1, 1, 0, bias=False),
            ))
        self.vq_input_proj = nn.ModuleList(vq_input_proj)

        vq_output_proj = []
        for i in range(self.num_vq):
            vq_output_proj.append(nn.Sequential(
                nn.Conv2d(self.hidden_dim + vq_embed_dims[i], self.hidden_dim, 1, 1, 0),
                # nn.ReLU(inplace=True)  # ORIGINALLY HERE
                # nn.LeakyReLU(0.1, inplace=True)
            ))
        self.vq_output_proj = nn.ModuleList(vq_output_proj)

        self.upsample = nn.ConvTranspose2d(
            vq_embed_dims[0], vq_embed_dims[0], 4, stride=2, padding=1
        )
        self.agg_type = cfg["vq"].get("agg_type", "concat")
        if (self.agg_type == "cat") or (self.agg_type == "concat"):
            self.agg_type = "concat"
            self.vq_aggregate_proj = nn.Conv2d(sum(vq_embed_dims), self.hidden_dim, 1, 1, 0)
        elif (self.agg_type == "add") or (self.agg_type == "sum"):
            self.agg_type = "add"
            self.vq_aggregate_proj = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1, 1, 0)
        else:
            raise ValueError(f"Unsupported aggregate type {self.agg_type}.")

        # -------- decoder -------- #
        num_dec_blocks = cfg["dec_num_blocks"]
        dec_proj = []
        for i in range(num_dec_blocks):
            dec_proj.append(
                DecResBlock(self.hidden_dim, self.feat_dim if (i == num_dec_blocks - 1) else self.hidden_dim))
        self.dec_proj = nn.Sequential(*dec_proj)

        last_norm = cfg.get("last_norm", False)
        self.dec_norm = LayerNorm2d(self.feat_dim) if last_norm else None

    def _photo_aug(self, x: torch.Tensor):
        # b, 3, h, w = x.shape
        batch_size = x.shape[0]
        device = x.device
        random_scale = torch.ones(batch_size, 3, 1, 1, dtype=torch.float32, device=device).uniform_(0.9,
                                                                                                    1.1)  # noqa # color
        random_offset = torch.ones(batch_size, 3, 1, 1, dtype=torch.float32, device=device).uniform_(-0.1, 0.1)  # noqa
        x_aug = x * random_scale + random_offset

        if random.randint(0, 3) == 0:  # 25%
            x_aug = transforms.GaussianBlur(kernel_size=3)(x_aug)  # texture
        return x_aug

    def forward(self, img: torch.Tensor
                ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:

        # photometric aug
        img_aug_1 = img
        img_aug_2 = self._photo_aug(img)

        img = torch.cat([img_aug_1, img_aug_2], dim=0)  # (2b, 3, h, w)

        dino_feat = self.extractor(img)  # (2b, 384, 28, 28)

        feat_bottom = self.enc_proj_bottom(dino_feat)
        feat_top = self.enc_proj_top(feat_bottom)

        print('feat_bottom : ', feat_bottom.shape)
        print("feat_top : ", feat_top.shape)

        output = dict()
        feat_vqs = []

        feat_0 = self.vq_input_proj[0](feat_top)

        feat_vq_0, vq_output_0, vq_top_dis_prob = self.vq_blocks[0](feat_0)
        print(feat_0.shape, feat_vq_0.shape, vq_top_dis_prob.shape)
        feat_vqs.append(feat_vq_0)

        for k, v in vq_output_0.items():
            output[f"vq{0}-{k}"] = v

        upsample_feat_vq_0 = self.upsample(feat_vq_0)
        print("upsample_feat_vq_0", upsample_feat_vq_0.shape)
        concat_feat = torch.cat([feat_bottom, upsample_feat_vq_0], dim=1)
        print("Concat_feat", concat_feat.shape)
        feat_1 = self.vq_output_proj[1](concat_feat)
        print("feat_1", feat_1.shape)
        feat_vq_1, vq_output_1, vq_bottom_dis_prob = self.vq_blocks[1](feat_1)
        print(feat_vq_1.shape, vq_bottom_dis_prob.shape)
        exit()
        feat_vqs.append(feat_vq_1)
        for k, v in vq_output_1.items():
            output[f"vq{1}-{k}"] = v

        if self.agg_type == "concat":
            feat = torch.cat(feat_vqs, dim=1)
        elif self.agg_type == "add":
            feat = sum(feat_vqs)
        else:
            raise ValueError
        feat = self.vq_aggregate_proj(feat)  # (b, 384, 28, 28)

        recon = self.dec_proj(feat)  # (b, 384, 28, 28)
        recon_loss = F.mse_loss(recon, dino_feat)

        output["recon-loss"] = recon_loss

        # contrastive loss
        top_dis_prob_1, top_dis_prob_2 = torch.chunk(vq_top_dis_prob, chunks=2, dim=0)  # (2bhw, K) -> (2, bhw, K)
        output["contra-loss-pos"] = self.jsd(top_dis_prob_1, top_dis_prob_2)

        bottom_dis_prob_1, bottom_dis_prob_2 = torch.chunk(vq_bottom_dis_prob, chunks=2, dim=0)
        output["contra-loss-neg"] = self.jsd(bottom_dis_prob_1, bottom_dis_prob_2)

        # TODO heuristic
        output["contra-loss"] = output["contra-loss-pos"] - output["contra-loss-neg"] * 0.01
        # output["contra-loss"] = output["contra-loss-pos"]

        # split half
        feat = torch.chunk(feat, chunks=2, dim=0)[0]
        feat_vqs = [torch.chunk(vq_i, chunks=2, dim=0)[0] for vq_i in feat_vqs]

        return feat, feat_vqs, output

    def restart(self):
        for i in range(self.num_vq):
            self.vq_blocks[i].restart()
