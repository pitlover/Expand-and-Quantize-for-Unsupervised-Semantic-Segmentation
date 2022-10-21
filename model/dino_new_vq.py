from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.dino import DinoFeaturizer
from model.blocks.resnet import EncResBlock, DecResBlock

from utils.dist_utils import all_reduce_tensor
import numpy as np
from sklearn.cluster import KMeans

class DINONewVq(nn.Module):
    def __init__(self, cfg: dict, cfg_loss: dict):
        super().__init__()
        self.cfg = cfg
        self.cfg_loss = cfg_loss

        self.extractor = DinoFeaturizer(cfg["pretrained"])
        self.feat_dim = self.extractor.n_feats  # 384
        self.hidden_dim = cfg.get("hidden_dim", self.feat_dim)

        # -------- semantic-encoder -------- #
        # num_enc_blocks = cfg["enc_num_blocks"]
        # semantic_enc_proj = []
        # for i in range(num_enc_blocks):
        #     semantic_enc_proj.append(EncResBlock(self.feat_dim if (i == 0) else self.hidden_dim, self.hidden_dim))
        # self.enc_proj = nn.Sequential(*semantic_enc_proj)

        # -------- vq -------- #
        vq_num_codebooks = cfg["vq"]["num_codebooks"]
        vq_embed_dims = cfg["vq"]["embed_dims"]
        assert len(vq_num_codebooks) == len(vq_embed_dims)
        self.num_vq = len(vq_num_codebooks)
        self.beta = cfg["vq"]["beta"]
        self.vq_type = cfg["vq"]["vq_type"]
        self.normalize = cfg["vq"].get("normalize", "none")
        self.use_weighted_sum = cfg["vq"].get("use_weighted_sum", False)
        self.need_initialized = cfg["vq"].get("need_initialized", False)

        vq_kwargs = dict(beta=self.beta,
                         normalize=self.normalize,
                         use_weighted_sum=self.use_weighted_sum,
                         need_initialized = self.need_initialized)

        self.num_pq = cfg["vq"].get("num_pq", 1)

        if isinstance(self.num_pq, int):
            self.num_pq = [self.num_pq] * self.num_vq

        if self.vq_type == "ema":
            raise ValueError("Not implemented")
            # vq_kwargs["decay"] = cfg["vq"]["decay"]
            # vq_kwargs["eps"] = cfg["vq"]["eps"]
            # vq_blocks = [
            #     EMAVectorQuantizer(vq_num_codebooks[i], vq_embed_dims[i], **vq_kwargs) if (self.num_pq == 1) else
            #     ProductQuantizerWrapper(self.num_pq[i], vq_num_codebooks[i], vq_embed_dims[i], **vq_kwargs,
            #                             quantizer_cls=EMAVectorQuantizer)
            #     for i in range(self.num_vq)
            # ]
        elif self.vq_type == "param":
            self.codebook = Codebook(num_codebook_vectors=vq_num_codebooks[0], latent_dim=vq_embed_dims[0], **vq_kwargs)
        else:
            raise ValueError(f"Unsupported vq type {self.vq_type}.")

        # -------- semantic-decoder -------- #
        num_dec_blocks = cfg["dec_num_blocks"]
        dec_proj = []
        for i in range(num_dec_blocks):
            dec_proj.append(
                DecResBlock(self.hidden_dim, self.feat_dim if (i == num_dec_blocks - 1) else self.hidden_dim))
        self.dec_proj = nn.Sequential(*dec_proj)

    def forward(self, img: torch.Tensor
                ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        # # photometric aug
        # img_aug_1 = img
        # img_aug_2 = self._photo_aug(img)
        #
        # img = torch.cat([img_aug_1, img_aug_2], dim=0)  # (2b, 3, h, w)

        dino_feat = self.extractor(img)  # (b, 384, 28, 28)
        feat = dino_feat
        # feat = self.enc_proj(dino_feat)  # (b, 384, 28, 28)

        output = dict()

        quantized_feat, vq_loss = self.codebook(feat)

        recon = self.dec_proj(quantized_feat)  # (2b, 384, 28, 28)
        recon_loss = F.mse_loss(recon, dino_feat)

        output["recon-loss"] = recon_loss
        output["vq-loss"] = vq_loss
        # # contrastive loss
        # top_dis_prob_1, top_dis_prob_2 = torch.chunk(vq_top_dis_prob, chunks=2, dim=0)  # (2bhw, K) -> (2, bhw, K)
        # output["contra-loss-pos"] = self.jsd(top_dis_prob_1, top_dis_prob_2)
        #
        # bottom_dis_prob_1, bottom_dis_prob_2 = torch.chunk(vq_bottom_dis_prob, chunks=2, dim=0)
        # output["contra-loss-neg"] = self.jsd(bottom_dis_prob_1, bottom_dis_prob_2)

        # split half
        # feat = torch.chunk(feat, chunks=2, dim=0)[0]
        # feat_vqs = [torch.chunk(vq_i, chunks=2, dim=0)[0] for vq_i in feat_vqs]
        return feat, quantized_feat, output


class Codebook(nn.Module):
    def __init__(self,
                 num_codebook_vectors: int,
                 latent_dim: int,
                 beta=0.25,
                 normalize: str = "none",
                 use_weighted_sum: bool = False,
                 need_initialized: str = "kmeans"):
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

        self.num_codebook_vectors = num_codebook_vectors
        self.embedding = nn.Embedding(num_codebook_vectors, latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_codebook_vectors, 1.0 / num_codebook_vectors)

        self.normalize = normalize
        if self.use_weighted_sum:
            assert self.normalize == "none", "Weight_sum should be unnormalized"
        if normalize == "z_trainable":
            self.z_mean = nn.Parameter(torch.zeros(self.embed_dim))
            self.z_log_var = nn.Parameter(torch.zeros(self.embed_dim))

        self.need_initialized = need_initialized

    def forward(self, z: torch.Tensor):
        z = z.permute(0, 2, 3, 1).contiguous()  # (b, d, h, w) -> (b, h, w, d)
        z_flat = z.view(-1, self.latent_dim)  # (bhw, d)

        if self.need_initialized != "none" and self.training:
            if self.need_initialized == "rand":
                self.prepare_restart(torch.zeros(self.num_codebook, dtype=torch.long, device=z.device),
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

            if self.training:
                with torch.no_grad():
                    z_flat_mean_orig = torch.mean(z_flat, dim=0)  # (d,)
                    z_flat_sq_mean_orig = torch.mean(z_flat * z_flat, dim=0)  # (d,)

                    z_flat_mean_orig = all_reduce_tensor(z_flat_mean_orig, op="mean")
                    z_flat_sq_mean_orig = all_reduce_tensor(z_flat_sq_mean_orig, op="mean")

                    z_flat_var_orig = z_flat_sq_mean_orig - (z_flat_mean_orig * z_flat_mean_orig)
                    z_flat_log_var_orig = z_flat_var_orig.log()

                    self.z_mean.data.mul_(self.decay).add_(z_flat_mean_orig, alpha=1 - self.decay)
                    self.z_log_var.data.mul_(self.decay).add_(z_flat_log_var_orig, alpha=1 - self.decay)

            z_norm = (z_flat - z_flat_mean) / (z_flat_std + 1e-5)
            # codebook_norm = (codebook - z_flat_mean) / (z_flat_std + 1e-5)

            codebook_std, codebook_mean = torch.std_mean(codebook, dim=0)  # (d,)
            codebook_norm = (codebook - codebook_mean) / (codebook_std + 1e-5)
        elif self.normalize == "none":
            z_norm = z_flat
            codebook_norm = codebook
        else:
            raise ValueError(f"Unsupported normalize type {self.normalize}")

        d = torch.sum(z_norm ** 2, dim=1, keepdim=True) + \
            torch.sum(codebook_norm ** 2, dim=1) - \
            2 * (torch.matmul(z_norm, codebook_norm.t()))

        min_encoding_indices = torch.argmin(d, dim=1)
        distance_prob = F.softmax(-d / 0.1, dim=1)

        if self.use_weighted_sum:
            z_q = torch.matmul(distance_prob , codebook_norm)  # TODO check temperature scaling
        else:
            z_q = self.embedding(min_encoding_indices)

        # compute loss for embedding
        codebook_loss = F.mse_loss(z_q, z_norm.detach())  # make codebook to be similar to input
        commitment_loss = F.mse_loss(z_norm, z_q.detach())  # make input to be similar to codebook
        q_loss = codebook_loss + self.beta * commitment_loss

        if not self.use_weighted_sum:
            z_q = z_norm + (z_q - z_norm).detach()
        z_q = z_q.view(z.shape).permute(0, 3, 1, 2).contiguous()

        return z_q, q_loss
