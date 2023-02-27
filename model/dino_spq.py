
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.dino import DinoFeaturizer
# TODO kmeans sampling
# from model.blocks.resnet_linear import EncResBlock, DecResBlock

# import faiss
from model.loss import InfoNCELoss, JSDLoss


class DINOSPQ(nn.Module):
    def __init__(self, cfg: dict, cfg_loss: dict):
        super().__init__()
        self.cfg = cfg
        self.cfg_loss = cfg_loss

        self.extractor = DinoFeaturizer(cfg["pretrained"])
        self.feat_dim = self.extractor.n_feats  # 384
        self.hidden_dim = cfg["vq"]["embed_dims"][0]
        # -------- semantic-encoder -------- #
        # num_enc_blocks = cfg["enc_num_blocks"]
        # semantic_enc_proj = []
        # for i in range(num_enc_blocks):
        #     semantic_enc_proj.append(EncResBlock(self.feat_dim if (i == 0) else self.hidden_dim, self.hidden_dim))
        # self.enc_proj = nn.Sequential(*semantic_enc_proj)
        self.enc_proj = nn.Conv2d(self.feat_dim, self.hidden_dim, (1, 1)) # TODO check

        # -------- vq -------- #
        vq_num_codebooks = cfg["vq"]["num_codebooks"]
        vq_embed_dims = cfg["vq"]["embed_dims"]
        assert len(vq_num_codebooks) == len(vq_embed_dims)
        self.vq_num_codebooks = vq_num_codebooks[0]
        self.num_vq = len(vq_num_codebooks)
        self.beta = cfg["vq"]["beta"]
        self.vq_type = cfg["vq"]["vq_type"]
        self.normalize = cfg["vq"].get("normalize", "none")
        self.use_weighted_sum = cfg["vq"].get("use_weighted_sum", False)
        self.use_restart = cfg["vq"].get("use_restart", False)
        self.need_initialized = cfg["vq"].get("need_initialized", False)
        self.pq_dropout = cfg["vq"].get("pq_dropout", 0.0)
        self.jsd_ts = cfg_loss["jsd"].get("temperature", 1.0)
        self.n_kmeans = cfg["vq"].get("n_kmeans", 1)
        vq_kwargs = dict(beta=self.beta,
                         normalize=self.normalize,
                         use_restart=self.use_restart,
                         use_weighted_sum=self.use_weighted_sum,
                         need_initialized=self.need_initialized,
                         pq_dropout=self.pq_dropout,
                         jsd_ts=self.jsd_ts)

        self.num_pq = cfg["vq"].get("num_pq", 1)

        if isinstance(self.num_pq, int):
            self.num_pq = [self.num_pq] * self.num_vq

        if self.vq_type == "ema":
            raise ValueError("Not implemented")

        elif self.vq_type == "param":
            vq_blocks = [
                ProductQuantizerWrapper(self.num_pq[i], vq_num_codebooks[i], vq_embed_dims[i])
                for i in range(self.num_vq)
            ]
            self.vq_blocks = nn.ModuleList(vq_blocks)
        else:
            raise ValueError(f"Unsupported vq type {self.vq_type}.")

        # -------- loss -------- #
        self.infonce_loss = InfoNCELoss(normalize=self.cfg_loss["info_nce"].get("normalize", "l2"),
                                        neg_sample=self.cfg_loss["info_nce"].get("neg_sample", 0),
                                        temperature=self.cfg_loss["info_nce"].get("temperature", 1.0),
                                        cal_type=self.cfg_loss["info_nce"].get("cal_type", "random")
                                        )

        # -------- final-linear -------- #

    def forward(self, img: torch.Tensor, aug_img: torch.Tensor = None, it: int = 0, stage: int = 0
                ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # photometric aug
        img = torch.cat([img, aug_img], dim=0)  # (2b, 3, h, w)

        dino_feat = self.extractor(img)  # (2b, 384, 28, 28)

        feat = self.enc_proj(dino_feat)  # (2b, hidden_dim, 28, 28)
        quantized_feat, outputs = self.vq_blocks[0](feat, it=it)  # (2b, hidden_dim, h, w)

        # MI loss
        semantic_feat_img1, semantic_feat_img2 = torch.chunk(feat, chunks=2, dim=0)  # (b, hidden_dim, h, w)
        outputs["info_nce"] = self.infonce_loss(semantic_feat_img1, semantic_feat_img2)
        # outputs["margin"] = self.margin_loss(semantic_feat_img1, semantic_feat_img2)

        # split half
        feat = torch.chunk(feat, chunks=2, dim=0)[0]
        quantized_feat = torch.chunk(quantized_feat, chunks=2, dim=0)[0]

        return feat, quantized_feat, outputs


def Soft_Quantization(X, C, N_books, tau_q):
    L_word = int(C.size()[1] // N_books)
    x = torch.split(X, L_word, dim=1)
    c = torch.split(C, L_word, dim=1)
    outputs = {"jsd": 0.0}
    jsd_loss = JSDLoss()
    for i in range(N_books):
        soft_c = F.softmax(squared_distances(x[i], c[i]) * (-tau_q), dim=-1)
        top_dis_prob_1, top_dis_prob_2 = torch.chunk(soft_c, chunks=2, dim=0)
        outputs["jsd"] += jsd_loss(top_dis_prob_1, top_dis_prob_2)
        if i == 0:
            Z = soft_c @ c[i]
        else:
            Z = torch.cat((Z, soft_c @ c[i]), dim=1)
    outputs["jsd"] /= N_books
    return Z, outputs


def squared_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    return torch.sum(diff * diff, -1)


class ProductQuantizerWrapper(nn.Module):

    def __init__(self,
                 num_pq: int,
                 num_codebook: int,
                 embed_dim: int,
                 ) -> None:
        super().__init__()
        if embed_dim % num_pq != 0:
            raise ValueError(f"Embed dim {embed_dim} should be divisible by #PQ {num_pq}.")
        self.num_codebook = num_codebook
        self.num_pq = num_pq
        self.pq_dim = embed_dim // num_pq

        # Codebooks
        self.C = torch.nn.Parameter((torch.randn(num_codebook, self.num_pq * self.pq_dim)).type(torch.float32),
                                    requires_grad=True)
        nn.init.xavier_uniform_(self.C)

        self.tau_q = 1.0  # TODO check

    def forward(self, z: torch.Tensor, it: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        b, c, h, w = z.shape
        z = z.permute(0, 2, 3, 1).contiguous()  # (b, d, h, w) -> (b, h, w, d)
        z_flat = z.view(-1, c)  # (bhw, d)

        z_quantized, outputs = Soft_Quantization(z_flat, self.C, self.num_pq, self.tau_q)
        z_quantized = z_quantized.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return z_quantized, outputs
