import os
from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from model.dino import DinoFeaturizer

from utils.dist_utils import all_reduce_tensor
import numpy as np
from sklearn.cluster import KMeans
from model.loss import InfoNCELoss, JSDLoss, JSDPosLoss, MarginRankingLoss, EntropyLoss, STEGOLoss
from model.blocks.module import SegmentationHead


class DINOPQGOCLS(nn.Module):
    def __init__(self, cfg: dict, cfg_loss: dict):
        super().__init__()
        self.cfg = cfg
        self.cfg_loss = cfg_loss

        self.extractor = DinoFeaturizer(cfg["pretrained"])
        self.feat_dim = self.extractor.n_feats  # 384
        self.hidden_dim = cfg["vq"]["embed_dims"][0]
        self.is_dropout = cfg["pretrained"]["dropout"]
        self.dropout = nn.Dropout2d(p=cfg["pretrained"]["drop_prob"])
        self.m = cfg["encoder"]["momentum"]  # TODO cosine annealing scheduling

        #  -------- student head -------- #
        self.b, self.d, self.h, self.w = None, None, None, None  # placeholder
        self.trainable_head = SegmentationHead(self.feat_dim, self.hidden_dim)

        #  -------- ema head -------- #
        self.ema_head = SegmentationHead(self.feat_dim, self.hidden_dim)

        for param_top, param_bottom in zip(self.trainable_head.parameters(), self.ema_head.parameters()):
            param_bottom.data.copy_(param_top.data)  # initialize
            param_bottom.requires_grad = False  # not update by gradient

        # # -------- vq -------- #
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
        self.use_split = cfg["vq"].get("use_split", False)
        self.need_initialized = cfg["vq"].get("need_initialized", False)
        self.pq_dropout = cfg["vq"].get("pq_dropout", 0.0)
        self.jsd_ts = cfg_loss["jsd"].get("temperature", 1.0)
        self.num_query = cfg_loss["jsd"].get("num_query", 3)
        self.num_pos = cfg_loss["jsd"].get("num_pos", 10)
        self.n_kmeans = cfg["vq"].get("n_kmeans", 1)
        vq_kwargs = dict(beta=self.beta,
                         normalize=self.normalize,
                         use_restart=self.use_restart,
                         use_split=self.use_split,
                         use_weighted_sum=self.use_weighted_sum,
                         need_initialized=self.need_initialized,
                         pq_dropout=self.pq_dropout,
                         jsd_ts=self.jsd_ts,
                         num_query=self.num_query,
                         num_pos=self.num_pos)

        self.num_pq = cfg["vq"].get("num_pq", 1)

        if isinstance(self.num_pq, int):
            self.num_pq = [self.num_pq] * self.num_vq

        vq_blocks = [
            Codebook(num_codebook_vectors=vq_num_codebooks[0], latent_dim=vq_embed_dims[0], **vq_kwargs)
            if (self.num_pq == 1) else
            ProductQuantizerWrapper(self.num_pq[i], vq_num_codebooks[i], vq_embed_dims[i], **vq_kwargs,
                                    quantizer_cls=Codebook)
            for i in range(self.num_vq)
        ]
        self.vq_blocks = nn.ModuleList(vq_blocks)

        # -------- classifier -------- #
        self.classifier = torch.nn.Linear(vq_embed_dims[0], self.num_pq[0] * vq_num_codebooks[0])

        # -------- loss -------- #
        self.stego_loss = STEGOLoss(cfg=self.cfg_loss["stego"])

    def _flatten(self, x):
        '''

        :param x: (b, d, h, w)
        :return:  (bhw, d)
        '''
        # b, d, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # (b, h, w, d)
        flat_x = x.view(-1, self.d)  # (bhw, d)

        return flat_x

    def _normalize(self, x: torch.Tensor):
        '''

        :param x: (b, d, h, w)
        :return: (bhw, d)
        '''

        x = self._flatten(x)
        norm_x = F.normalize(x, dim=-1)

        return norm_x

    @torch.no_grad()
    def _momentum_update_ema_head(self):
        """
        Momentum update of the ema encoder
        """
        for param_q, param_k in zip(self.trainable_head.parameters(), self.ema_head.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, img: torch.Tensor,
                aug_img: torch.Tensor = None,
                img_pos: torch.Tensor = None,
                it: int = 0, stage: int = 0
                ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        # photometric aug
        # for stego...
        outputs = {}

        dino_feat = self.extractor(img)  # (b, 384, 28, 28) (b, d, h, w)
        dino_feat = self.dropout(dino_feat)

        dino_feat_aug = self.extractor(aug_img)
        dino_feat_aug = self.dropout(dino_feat_aug)

        if self.training:
            dino_feat_pos = self.extractor(img_pos)  # (b, 384, 28, 28) (b, d, h, w)
            dino_feat_pos = self.dropout(dino_feat_pos)
            code_pos = self.trainable_head(dino_feat_pos)
        else:
            code_pos = None  # placeholder

        # ---- trainable(ori) vs ema(aug) ---- #
        z1_1 = self.trainable_head(dino_feat) # (b, d, h, w)
        self.b, self.d, self.h, self.w = z1_1.shape
        norm_z1_1 = self._normalize(z1_1)

        with torch.no_grad():
            self._momentum_update_ema_head()
            z1_2 = self.ema_head(dino_feat_aug)
            norm_z1_2 = self._normalize(z1_2).clone().detach()  # detach the gradient z2

        loss1 = F.mse_loss(norm_z1_1, norm_z1_2)
        outputs["mse-loss"] = loss1

        quantized_feat, outputs, distance_prob, pseudo_labels = self.vq_blocks[0](z1_2)  # (2b, hidden_dim, h, w)
        # recon = self.dec_proj(quantized_feat)

        # TODO vq part
        if self.training:
            # TODO vq part -> need remove
            outputs["stego-loss"] = self.stego_loss(dino_feat, dino_feat_pos, z1_1, code_pos)

        # classifier
        total_logits = self.classifier(self._flatten(z1_1)) # (bhw, 16384)
        print(total_logits.shape)
        print(len(pseudo_labels))
        exit()
        out = self._reshape(z1_1)

        return out, quantized_feat, outputs


class Codebook(nn.Module):
    def __init__(self,
                 num_codebook_vectors: int,
                 latent_dim: int,
                 beta=0.25,
                 normalize: str = "none",
                 use_restart: bool = False,
                 use_split: bool = False,
                 use_weighted_sum: bool = False,
                 need_initialized: str = "kmeans",
                 pq_dropout: float = 0.0,
                 jsd_ts: float = 1.0,
                 num_query: int = 3,
                 num_pos: int = 10
                 ):
        super(Codebook, self).__init__()
        """
        embedding: (num_vq, embed_dim)
        beta: the factor for vq_loss
        input: (b, embed_dim, h, w)
        output: (b, embed_dim, h, w)
        """
        # self.num_codebook_vectors = args.num_codebook_vectors
        self.latent_dim = latent_dim
        self.beta = beta
        self.use_weighted_sum = use_weighted_sum
        self.pq_dropout = pq_dropout
        self.num_codebook_vectors = num_codebook_vectors
        self.embedding = nn.Embedding(num_codebook_vectors, latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_codebook_vectors, 1.0 / num_codebook_vectors)
        self.vq_count = torch.zeros(self.num_codebook_vectors)

        self.update_indices = None  # placeholder
        self.update_candidates = None  # placeholder

        self.normalize = normalize
        if self.use_weighted_sum:
            assert self.normalize == "none", "Weight_sum should be unnormalized"
        if normalize == "z_trainable":
            self.z_mean = nn.Parameter(torch.zeros(self.latent_dim))
            self.z_log_var = nn.Parameter(torch.zeros(self.latent_dim))
        self.use_restart = use_restart
        self.use_split = use_split
        self.need_initialized = need_initialized
        # ----- JSD ------ #
        self.posjsd_loss = JSDPosLoss(num_query=num_query, num_pos=num_pos)
        self.jsd_ts = jsd_ts

    @torch.no_grad()
    def prepare_restart(self, vq_current_count: torch.Tensor, z_flat: torch.Tensor) -> None:
        """Restart dead entries
        :param vq_current_count:        (K,)
        :param z_flat:                  (n, d) = (bhw, d)
        """
        n_data = z_flat.shape[0]
        update_indices = torch.nonzero(vq_current_count == 0, as_tuple=True)[0]
        n_update = len(update_indices)

        z_indices = list(range(n_data))
        random.shuffle(z_indices)

        if n_update <= n_data:
            z_indices = z_indices[:n_update]
        else:
            update_indices = update_indices.tolist()
            random.shuffle(update_indices)
            update_indices = update_indices[:n_data]

        self.update_indices = update_indices
        self.update_candidates = z_flat[z_indices]

    @torch.no_grad()
    def restart(self) -> None:
        if (self.update_indices is not None) and (self.update_candidates is not None):
            self.embedding.weight.data[self.update_indices] = self.update_candidates.float()
            self.vq_count.fill_(0)

            self.update_indices = None
            self.update_candidates = None

    @torch.no_grad()
    def split(self, vq_current_count: torch.Tensor) -> int:
        """Split replace dead entries
        :param vq_current_count:        (K,)
        :return:                        codebooks that are not used
        """
        update_indices = torch.nonzero(vq_current_count == 0, as_tuple=True)[0]
        if len(update_indices) == 0:
            return 0
        n_update = len(update_indices)
        update_indices = update_indices[torch.randperm(n_update)]

        vq_total_count = self.vq_count  # (K,)
        vq_weight = self.embedding.weight  # (K, d)

        _, vq_total_count_sort = torch.sort(vq_total_count, dim=0, descending=True)
        vq_total_count_sort = vq_total_count_sort[:n_update]

        vq_total_count_candidate = vq_total_count.detach().clone()
        vq_weight_candidate = vq_weight.detach().clone()

        noise = torch.randn(n_update, self.latent_dim, dtype=vq_weight.dtype, device=vq_weight.device).mul_(0.02)
        vq_weight_candidate[update_indices] = vq_weight[vq_total_count_sort] + noise
        vq_weight_candidate[vq_total_count_sort] = vq_weight[vq_total_count_sort] - noise
        vq_total_count_candidate[update_indices] = vq_total_count[vq_total_count_sort] / 2.0
        vq_total_count_candidate[vq_total_count_sort] = vq_total_count[vq_total_count_sort] / 2.0

        self.vq_count.data.copy_(vq_total_count_candidate.data)
        self.embedding.weight.data.copy_(vq_weight_candidate.data)

        self.vq_count.fill_(0)
        return n_update

    def forward(self, z: torch.Tensor):  # i-th pq, iteration
        output = dict()
        self.vq_count = self.vq_count.to(z.device)
        b, d, h, w = z.shape
        z = z.permute(0, 2, 3, 1).contiguous()  # (b, d, h, w) -> (b, h, w, d)
        z_flat = z.view(-1, self.latent_dim)  # (bhw, d)

        if self.need_initialized != "none" and self.training:
            if self.need_initialized == "rand":
                self.prepare_restart(torch.zeros(self.num_codebook_vectors, dtype=torch.long, device=z.device),
                                     z_flat)
                self.restart()

            elif self.need_initialized == "kmeans":
                clustering = KMeans(init="k-means++", n_clusters=self.num_codebook_vectors, random_state=0)
                cpu_z_flat = z_flat.detach().cpu().numpy()
                clustering.fit(cpu_z_flat)
                centroids = np.array(clustering.cluster_centers_)
                centroids = torch.from_numpy(centroids).float().to(z.device)
                self.embedding.weight.data.copy_(centroids)

            elif self.need_initialized == "uni":
                nn.init.xavier_uniform_(self.embedding.weight)

            elif self.need_initialized == "normal":
                nn.init.xavier_normal_(self.embedding.weight)

            self.need_initialized = "none"

        codebook = self.embedding.weight

        if self.normalize == "l2":
            z_norm = F.normalize(z_flat, dim=1)
            codebook_norm = F.normalize(codebook, dim=1)

        elif self.normalize == "z_norm":  # z-normalize
            z_flat_std, z_flat_mean = torch.std_mean(z_flat, dim=1, keepdim=True)  # (n, 1)
            z_norm = (z_flat - z_flat_mean) / (z_flat_std + 1e-5)

            codebook_std, codebook_mean = torch.std_mean(codebook, dim=1, keepdim=True)  # (K, 1)
            codebook_norm = (codebook - codebook_mean) / (codebook_std + 1e-5)
        elif self.normalize == "z_trainable":
            z_flat_mean = self.z_mean
            z_flat_std = self.z_log_var.exp().sqrt()

            z_norm = (z_flat - z_flat_mean) / (z_flat_std + 1e-5)

            codebook_std, codebook_mean = torch.std_mean(codebook, dim=0)  # (d,)
            codebook_norm = (codebook - codebook_mean) / (codebook_std + 1e-5)
        elif self.normalize == "none":
            z_norm = z_flat
            codebook_norm = codebook
        else:
            raise ValueError(f"Unsupported normalize type {self.normalize}")
        # pq_dropout
        if self.pq_dropout > 0.0:
            pq_mask = torch.cuda.FloatTensor(self.num_codebook_vectors).uniform_() > self.pq_dropout
            codebook_norm = codebook_norm[pq_mask]

        d = torch.sum(z_norm ** 2, dim=1, keepdim=True) + \
            torch.sum(codebook_norm ** 2, dim=1) - \
            2 * (torch.matmul(z_norm, codebook_norm.t()))  # (2bhw, n_prototypes)

        min_encoding_indices = torch.argmin(d, dim=1)
        distance_prob = F.softmax(-d / self.jsd_ts, dim=1)  # (2bhw, n_prototypes)
        vq_indices = torch.argmin(d, dim=1)
        if self.use_weighted_sum:
            z_q = torch.matmul(distance_prob, codebook_norm)  # TODO check temperature scaling
        else:
            z_q = self.embedding(min_encoding_indices)

        # avoid collapse
        if self.training:
            with torch.no_grad():
                vq_indices_one_hot = F.one_hot(vq_indices, self.num_codebook_vectors).to(z.dtype)  # (n, K)

                vq_current_count = torch.sum(vq_indices_one_hot, dim=0)  # (K,)
                vq_current_count = all_reduce_tensor(vq_current_count, op="sum")

                self.vq_count += vq_current_count

                if self.use_restart:
                    self.prepare_restart(vq_current_count, z_norm)
                    self.restart()

                output["codebook-usage"] = (codebook_norm.shape[0] - len(
                    torch.nonzero(vq_current_count == 0, as_tuple=True)[0])) / codebook_norm.shape[0]  # used ratio

        # compute loss for embedding
        codebook_loss = F.mse_loss(z_q, z_norm.detach())  # make codebook to be similar to input
        commitment_loss = F.mse_loss(z_norm, z_q.detach())  # make input to be similar to codebook
        q_loss = codebook_loss + self.beta * commitment_loss

        if not self.use_weighted_sum:
            z_q = z_norm + (z_q - z_norm).detach()

        # TODO kmeans sampling
        z_q = z_q.view(z.shape).permute(0, 3, 1, 2).contiguous()
        distance_prob = distance_prob.view(b, h, w, -1).contiguous()
        output["vq-loss"] = q_loss

        return z_q, output, distance_prob, min_encoding_indices


class ProductQuantizerWrapper(nn.Module):

    def __init__(self,
                 num_pq: int,
                 num_codebook: int,
                 embed_dim: int,
                 beta: float = 0.25,  # commitment loss
                 normalize: Optional[str] = None,
                 decay: float = 0.99,
                 eps: float = 1e-5,
                 use_restart: bool = False,
                 use_split: bool = False,
                 use_gumbel: bool = False,
                 use_weighted_sum: bool = False,
                 update_norm: bool = True,
                 need_initialized: str = "none",
                 pq_dropout: float = 0.0,
                 jsd_ts: float = 1.0,
                 num_query: int = 3,
                 num_pos: int = 10,
                 quantizer_cls=Codebook,
                 ) -> None:
        super().__init__()
        if embed_dim % num_pq != 0:
            raise ValueError(f"Embed dim {embed_dim} should be divisible by #PQ {num_pq}.")
        self.num_pq = num_pq
        self.pq_dim = embed_dim // num_pq

        self.quantizers = nn.ModuleList([
            quantizer_cls(num_codebook, self.pq_dim, beta=beta, normalize=normalize, use_restart=use_restart,
                          use_split=use_split,
                          use_weighted_sum=use_weighted_sum, need_initialized=need_initialized, pq_dropout=pq_dropout,
                          jsd_ts=jsd_ts, num_query=num_query, num_pos=num_pos)
            for _ in range(self.num_pq)
        ])

    def forward(self, z: torch.Tensor, it: int = -1) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        # b, c, h, w = z.shape
        z_split = torch.chunk(z, chunks=self.num_pq, dim=1)
        z_quantized = list()
        outputs = dict()
        distance_prob = list()
        pseudo_labels = list()

        for i in range(self.num_pq):
            q_i, output_i, prob_i, pseudo_label = self.quantizers[i](
                z_split[i])  # (2bhw, dim // n_prototypes) -> (2b, dim // n_prototypes, h, w)
            z_quantized.append(q_i)
            if i == 0:
                for k, v in output_i.items():
                    outputs[k] = v
            else:
                for k, v in output_i.items():
                    outputs[k] = outputs[k] + v
            distance_prob.append(prob_i)  # (2bhw, n_prototypes)
            pseudo_labels.append(pseudo_label)
        z_quantized = torch.cat(z_quantized, dim=1)
        for k, v in outputs.items():
            outputs[k] /= self.num_pq

        distance_prob = torch.cat(distance_prob, dim=-1)  # (2bhw, n_prototypes x #pq)

        return z_quantized, outputs, distance_prob, pseudo_labels
