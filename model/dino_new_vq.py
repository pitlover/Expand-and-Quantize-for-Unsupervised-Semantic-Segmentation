from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from model.dino import DinoFeaturizer
from model.blocks.resnet import EncResBlock, DecResBlock

from utils.dist_utils import all_reduce_tensor
import numpy as np
from sklearn.cluster import KMeans
import faiss


@torch.no_grad()
def get_histogram_count(count: torch.Tensor, prefix: str = "") -> Dict:
    prob = count.float() / (count.sum() + 1)  # (K,)
    num_codebook = len(prob)

    prob, _ = torch.sort(prob, dim=0, descending=True)  # (K,)
    c_sum = torch.cumsum(prob, dim=0)  # (K,)
    output = {f"{prefix}-p10": None, f"{prefix}-p50": None, f"{prefix}-p90": None}
    for i in range(len(c_sum)):
        if (c_sum[i] >= 0.9) and (output[f"{prefix}-p90"] is None):
            output[f"{prefix}-p90"] = i / num_codebook
        if (c_sum[i] >= 0.5) and (output[f"{prefix}-p50"] is None):
            output[f"{prefix}-p50"] = i / num_codebook
        if (c_sum[i] >= 0.1) and (output[f"{prefix}-p10"] is None):
            output[f"{prefix}-p10"] = i / num_codebook
    return output


class DINONewVq(nn.Module):
    def __init__(self, cfg: dict, cfg_loss: dict):
        super().__init__()
        self.cfg = cfg
        self.cfg_loss = cfg_loss

        self.extractor = DinoFeaturizer(cfg["pretrained"])
        self.feat_dim = self.extractor.n_feats  # 384
        self.hidden_dim = cfg.get("hidden_dim", self.feat_dim)

        # -------- semantic-encoder -------- #
        num_enc_blocks = cfg["enc_num_blocks"]
        semantic_enc_proj = []
        for i in range(num_enc_blocks):
            semantic_enc_proj.append(EncResBlock(self.feat_dim if (i == 0) else self.hidden_dim, self.hidden_dim))
        self.enc_proj = nn.Sequential(*semantic_enc_proj)

        # -------- vq -------- #
        vq_num_codebooks = cfg["vq"]["num_codebooks"]
        vq_embed_dims = cfg["vq"]["embed_dims"]
        assert len(vq_num_codebooks) == len(vq_embed_dims)
        self.num_vq = len(vq_num_codebooks)
        self.beta = cfg["vq"]["beta"]
        self.vq_type = cfg["vq"]["vq_type"]
        self.normalize = cfg["vq"].get("normalize", "none")
        self.use_weighted_sum = cfg["vq"].get("use_weighted_sum", False)
        self.use_restart = cfg["vq"].get("use_restart", False)
        self.need_initialized = cfg["vq"].get("need_initialized", False)

        vq_kwargs = dict(beta=self.beta,
                         normalize=self.normalize,
                         use_restart=self.use_restart,
                         use_weighted_sum=self.use_weighted_sum,
                         need_initialized=self.need_initialized)

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
            vq_blocks = [
                Codebook(num_codebook_vectors=vq_num_codebooks[0], latent_dim=vq_embed_dims[0], **vq_kwargs)
                if (self.num_pq == 1) else
                ProductQuantizerWrapper(self.num_pq[i], vq_num_codebooks[i], vq_embed_dims[i], **vq_kwargs,
                                        quantizer_cls=Codebook)
                for i in range(self.num_vq)
            ]
            self.vq_blocks = nn.ModuleList(vq_blocks)
            # self.codebook = Codebook(num_codebook_vectors=vq_num_codebooks[0], latent_dim=vq_embed_dims[0], **vq_kwargs)
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
        feat = self.enc_proj(dino_feat)  # (b, 384, 28, 28)

        quantized_feat, outputs, distance_prob = self.vq_blocks[0](feat)

        recon = self.dec_proj(quantized_feat)  # (2b, 384, 28, 28)
        recon_loss = F.mse_loss(recon, dino_feat)

        outputs["recon-loss"] = recon_loss

        # # contrastive loss
        # top_dis_prob_1, top_dis_prob_2 = torch.chunk(vq_top_dis_prob, chunks=2, dim=0)  # (2bhw, K) -> (2, bhw, K)
        # output["contra-loss-pos"] = self.jsd(top_dis_prob_1, top_dis_prob_2)
        #
        # bottom_dis_prob_1, bottom_dis_prob_2 = torch.chunk(vq_bottom_dis_prob, chunks=2, dim=0)
        # output["contra-loss-neg"] = self.jsd(bottom_dis_prob_1, bottom_dis_prob_2)

        # split half
        # feat = torch.chunk(feat, chunks=2, dim=0)[0]
        # feat_vqs = [torch.chunk(vq_i, chunks=2, dim=0)[0] for vq_i in feat_vqs]
        return feat, quantized_feat, outputs


class Codebook(nn.Module):
    def __init__(self,
                 num_codebook_vectors: int,
                 latent_dim: int,
                 beta=0.25,
                 normalize: str = "none",
                 use_restart: bool = False,
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
        self.need_initialized = need_initialized

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
            self.embedding.weight.data[self.update_indices].copy_(self.update_candidates)

            self.vq_count.fill_(0)

            self.update_indices = None
            self.update_candidates = None

    def forward(self, z: torch.Tensor):
        output = dict()

        self.vq_count = self.vq_count.to(z.device)

        z = z.permute(0, 2, 3, 1).contiguous()  # (b, d, h, w) -> (b, h, w, d)
        z_flat = z.view(-1, self.latent_dim)  # (bhw, d)
        if self.need_initialized != "none" and self.training:
            if self.need_initialized == "rand":
                print("Random Initializing ... ")
                self.prepare_restart(torch.zeros(self.num_codebook_vectors, dtype=torch.long, device=z.device),
                                     z_flat)
                self.restart()

            elif self.need_initialized == "kmeans":
                print("Kmeans Initializing ... ")
                # clustering = KMeans(init="k-means++", n_clusters=self.num_codebook_vectors, random_state=0)
                # cpu_z_flat = z_flat.detach().cpu().numpy()
                # clustering.fit(cpu_z_flat)
                # centroids = np.array(clustering.cluster_centers_)
                # centroids = torch.from_numpy(centroids).float().to(z.device)
                # self.embedding.weight.data.copy_(centroids)
                kmeans = faiss.Kmeans(d=z_flat.shape[-1], k=self.num_codebook_vectors, verbose=True, gpu=True)
                cpu_z_flat = z_flat.detach().cpu().numpy().astype(np.float32)
                kmeans.train(cpu_z_flat)
                centroids = torch.from_numpy(kmeans.centroids).float().to(z.device)
                self.embedding.weight.data.copy_(centroids)

            elif self.need_initialized == "uni":
                print("Uni Initializing ... ")
                nn.init.xavier_uniform_(self.embedding.weight)

            elif self.need_initialized == "normal":
                print("Normal Initializing ... ")
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

        d = torch.sum(z_norm ** 2, dim=1, keepdim=True) + \
            torch.sum(codebook_norm ** 2, dim=1) - \
            2 * (torch.matmul(z_norm, codebook_norm.t()))

        min_encoding_indices = torch.argmin(d, dim=1)
        distance_prob = F.softmax(-d / 0.5, dim=1)
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

                # vq_current_hist = get_histogram_count(vq_current_count, prefix="current")
                # vq_hist = get_histogram_count(self.vq_count, prefix="total")
                # output.update(vq_hist)
                # output.update(vq_current_hist)

                if self.use_restart:
                    self.prepare_restart(vq_current_count, z_flat)

                # if self.use_split:
                #     n_split = self.split(vq_current_count)  # not-used count
                # else:
                #     n_split = len(torch.nonzero(vq_current_count == 0, as_tuple=True)[0])
                output["codebook-usage"] = (self.num_codebook_vectors - len(
                    torch.nonzero(vq_current_count == 0, as_tuple=True)[0])) / self.num_codebook_vectors  # used ratio

        # compute loss for embedding
        codebook_loss = F.mse_loss(z_q, z_norm.detach())  # make codebook to be similar to input
        commitment_loss = F.mse_loss(z_norm, z_q.detach())  # make input to be similar to codebook
        q_loss = codebook_loss + self.beta * commitment_loss

        if not self.use_weighted_sum:
            z_q = z_norm + (z_q - z_norm).detach()
        z_q = z_q.view(z.shape).permute(0, 3, 1, 2).contiguous()

        output["vq-loss"] = q_loss

        return z_q, output, distance_prob


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
                 use_gumbel: bool = False,
                 use_split: bool = False,
                 use_weighted_sum: bool = False,
                 update_norm: bool = True,
                 need_initialized: str = "none",
                 quantizer_cls=Codebook,
                 ) -> None:
        super().__init__()
        if embed_dim % num_pq != 0:
            raise ValueError(f"Embed dim {embed_dim} should be divisible by #PQ {num_pq}.")
        self.num_pq = num_pq
        self.pq_dim = embed_dim // num_pq

        self.quantizers = nn.ModuleList([
            quantizer_cls(num_codebook, self.pq_dim, beta=beta, normalize=normalize, use_restart=use_restart,
                          use_weighted_sum=use_weighted_sum, need_initialized=need_initialized)
            for _ in range(self.num_pq)
        ])

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        # b, c, h, w = z.shape
        z_split = torch.chunk(z, chunks=self.num_pq, dim=1)

        z_quantized = list()
        outputs = dict()
        distance_prob = list()

        for i in range(self.num_pq):
            q_i, output_i, prob_i = self.quantizers[i](z_split[i])
            z_quantized.append(q_i)
            if i == 0:
                for k, v in output_i.items():
                    outputs[k] = v
            else:
                for k, v in output_i.items():
                    outputs[k] = outputs[k] + v
            distance_prob.append(prob_i)

        z_quantized = torch.cat(z_quantized, dim=1)

        for k, v in outputs.items():
            outputs[k] /= self.num_pq

        distance_prob = torch.cat(distance_prob, dim=-1)  # (n, K x #pq)

        return z_quantized, outputs, distance_prob
