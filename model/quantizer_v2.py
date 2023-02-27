"""https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py"""
import random
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from utils.dist_utils import all_reduce_tensor, broadcast_tensors

__all__ = ["VectorQuantizer", "EMAVectorQuantizer", "ProductQuantizerWrapper"]


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


class VectorQuantizer(nn.Module):

    def __init__(self,
                 num_codebook: int,
                 embed_dim: int,
                 beta: float = 0.25,  # commitment loss
                 normalize: Optional[str] = None,
                 use_restart: bool = False,
                 use_gumbel: bool = False,
                 use_split: bool = False,
                 use_weighted_sum: bool = False,
                 update_norm: bool = True
                 ) -> None:
        super().__init__()
        self.num_codebook = num_codebook
        self.embed_dim = embed_dim
        self.beta = beta
        self.normalize = normalize

        # if normalize == "z_trainable":
        #     self.z_mean = nn.Parameter(torch.zeros(self.embed_dim))
        #     self.z_log_var = nn.Parameter(torch.zeros(self.embed_dim))

        self.codebook = nn.Embedding(self.num_codebook, self.embed_dim)  # codebook: (K, d)
        # self.codebook.weight.data.uniform_(-1.0 / self.num_codebook, 1.0 / self.num_codebook)  # TODO initialize diff?
        nn.init.xavier_normal_(self.codebook.weight)

        self.register_buffer("vq_count", torch.zeros(self.num_codebook), persistent=False)
        self.use_restart = use_restart
        self.use_gumbel = use_gumbel
        self.use_split = use_split
        self.use_weighted_sum = use_weighted_sum
        self.update_norm = update_norm
        if use_split:
            raise NotImplementedError("NOT YET implemented. Currently only for EMA.")

        self.update_indices = None  # placeholder
        self.update_candidates = None  # placeholder

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
            self.codebook.weight.data[self.update_indices].copy_(self.update_candidates)
            self.vq_count.fill_(0)

            self.update_indices = None
            self.update_candidates = None

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """VQ forward
        :param z:       (batch_size, embed_dim, h, w)
        :return:        (batch_size, embed_dim, h, w) quantized
                        loss = codebook_loss + beta * commitment_loss
                        statistics
        """
        z = z.permute(0, 2, 3, 1).contiguous()
        b, h, w, d = z.shape

        z_flat = z.view(-1, d)  # (bhw, d) = (n, d)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        codebook = self.codebook.weight  # (K, d)

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
            codebook_norm = (codebook - z_flat_mean) / (z_flat_std + 1e-5)
        elif self.normalize == "none":
            z_norm = z_flat
            codebook_norm = codebook
        else:
            raise ValueError(f"Unsupported normalize type {self.normalize}")

        distance = (
                torch.sum(z_norm ** 2, dim=1, keepdim=True) +  # (n, d) -> (n, 1)
                torch.sum(codebook_norm ** 2, dim=1) -  # (K, d) -> (K,) == (1, K)
                2 * torch.matmul(z_norm, codebook_norm.t())  # (n, K)
        )  # (n, K)

        if self.training and self.use_gumbel:
            distance_prob_gumbel = F.gumbel_softmax(-distance, tau=1.0, hard=True, dim=1)
            vq_indices = torch.argmax(distance_prob_gumbel, dim=1)
        else:
            vq_indices = torch.argmin(distance, dim=1)  # (n,) : index of the closest code vector.
        distance_prob = F.softmax(-distance, dim=1)
        # z_quantized = self.codebook(vq_indices)  # (n, d)
        z_norm_quantized = F.embedding(vq_indices, codebook_norm)  # (n, d)

        output = dict()

        # update count
        with torch.no_grad():
            vq_indices_one_hot = F.one_hot(vq_indices, self.num_codebook)  # (n, K)
            vq_current_count = torch.sum(vq_indices_one_hot, dim=0)  # (K,)

            vq_current_count = all_reduce_tensor(vq_current_count, op="sum")

            self.vq_count += vq_current_count

            vq_current_hist = get_histogram_count(vq_current_count, prefix="current")
            vq_hist = get_histogram_count(self.vq_count, prefix="total")
            output.update(vq_hist)
            output.update(vq_current_hist)

            if self.use_restart:
                self.prepare_restart(vq_current_count, z_flat)

        # compute loss for embedding
        codebook_loss = F.mse_loss(z_norm_quantized, z_norm.detach())  # make codebook to be similar to input
        commitment_loss = F.mse_loss(z_norm, z_norm_quantized.detach())  # make input to be similar to codebook
        loss = codebook_loss + self.beta * commitment_loss

        output["loss"] = loss
        output["codebook_loss"] = codebook_loss
        output["commitment_loss"] = commitment_loss

        # preserve gradients
        z_norm_quantized = z_norm + (z_norm_quantized - z_norm).detach()  # (n, d)

        # reshape back to match original input shape
        q = z_norm_quantized.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()

        return q, output, distance_prob

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, " \
               f"num_codebook={self.num_codebook}, " \
               f"normalize={self.normalize}, " \
               f"use_gumbel={self.use_gumbel}, " \
               f"use_split={self.use_split}, " \
               f"use_weighted_sum={self.use_weighted_sum}, " \
               f"update_norm={self.update_norm}, " \
               f"use_restart={self.use_restart}"


class EMAVectorQuantizer(nn.Module):
    def __init__(self,
                 n_codes: int,
                 embedding_dim: int,
                 beta: float = 0.25,  # commitment loss
                 normalize: Optional[str] = None,
                 decay: float = 0.99,
                 eps: float = 1e-5,
                 use_restart: bool = False,
                 use_gumbel: bool = False,
                 use_split: bool = False,
                 use_weighted_sum: bool = False,
                 update_norm: bool = True
                 ) -> None:
        super().__init__()
        weight = torch.randn(n_codes, embedding_dim)
        nn.init.uniform_(weight, -1.0 / n_codes, 1.0 / n_codes)

        self.register_buffer('embeddings', weight)
        self.register_buffer('N', torch.zeros(n_codes))
        self.register_buffer('z_avg', self.embeddings.data.clone())

        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self._need_init = False

        self.decay = decay
        self.eps = eps
        self.beta = beta

    def _tile(self, x):
        n_data = x.shape[0]
        update_indices = torch.zeros(self.n_codes, dtype=torch.long, device=x.device)
        n_update = len(update_indices)

        z_indices = list(range(n_data))

        if n_update <= n_data:
            z_indices = z_indices[:n_update]
        else:
            raise ValueError("n_update > n_data")

        update_candidates = x[z_indices]

        return update_candidates

    def _init_embeddings(self, flat_inputs):
        # flat_inputs: [bhw, d]
        self._need_init = False
        _k_rand = self._tile(flat_inputs)
        broadcast_tensors(_k_rand, 0)
        self.embeddings.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.n_codes))

    def forward(self, z):
        # z: [b, d, h, w]
        z = z.permute(0, 2, 3, 1).contiguous()  # (b, h, w, d)
        b, h, w, d = z.shape

        flat_inputs = z.view(-1, d)  # (bhw, d) = (n, d)

        if self._need_init and self.training:
            self._init_embeddings(flat_inputs)

        # normalize
        z_norm = F.normalize(flat_inputs, dim=1)
        codebook_norm = F.normalize(self.embeddings, dim=1)
        flat_inputs = z_norm
        distances = (torch.sum(z_norm ** 2, dim=1, keepdim=True)
                     + torch.sum(codebook_norm ** 2, dim=1)
                     - 2 * torch.matmul(z_norm, codebook_norm.t()))

        distance_prob = F.softmax(-distances * 1.0, dim=1)  # (n, K) # TODO scaling

        encoding_indices = torch.argmin(distances, dim=1)
        embeddings = F.embedding(encoding_indices, z_norm)
        output = dict()

        # EMA codebook update
        if self.training:
            encode_onehot = F.one_hot(encoding_indices, self.n_codes).to(z.dtype)
            n_total = encode_onehot.sum(dim=0)
            encode_sum = torch.matmul(encode_onehot.t(), flat_inputs)
            # encode_sum = flat_inputs.t() @ encode_onehot # (n, d) -> (d, n) * (n, k) -> (d, k)
            all_reduce_tensor(n_total)
            all_reduce_tensor(encode_sum)

            self.N.data.mul_(self.decay).add_(n_total, alpha=1 - self.decay)
            self.z_avg.data.mul_(self.decay).add_(encode_sum, alpha=1 - self.decay)

            n = self.N.sum()
            weights = (self.N + self.eps) / (n + self.n_codes * self.eps) * n
            encode_normalized = self.z_avg / weights.unsqueeze(1)
            self.embeddings.data.copy_(encode_normalized)

            # _k_rand = self._tile(flat_inputs)
            # broadcast_tensors(_k_rand, 0)
            # usage = (self.N.reshape(self.n_codes, 1) >= 1).float()
            # self.embeddings.data.mul_(usage).add_(_k_rand * (1 - usage))

        commitment_loss = F.mse_loss(flat_inputs, embeddings.detach())

        output["commitment-loss"] = commitment_loss
        output["loss"] = self.beta * commitment_loss

        output["codebook-sum"] = torch.sum(torch.abs(self.embeddings))

        q = embeddings.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()  # (b, d, h, w)

        return q, output, distance_prob


# class EMAVectorQuantizer(nn.Module):
#     def __init__(self,
#                  num_codebook: int,
#                  embed_dim: int,
#                  beta: float = 0.25,  # commitment loss
#                  normalize: Optional[str] = None,
#                  decay: float = 0.99,
#                  eps: float = 1e-5,
#                  use_restart: bool = False,
#                  use_gumbel: bool = False,
#                  use_split: bool = False,
#                  use_weighted_sum: bool = False,
#                  update_norm: bool = True
#                  ) -> None:
#         super().__init__()
#         self.num_codebook = num_codebook
#         self.embed_dim = embed_dim
#         self.beta = beta
#         self.decay = decay
#         self.normalize = normalize
#
#         # if normalize == "z_trainable":
#         #     self.register_buffer("z_mean", torch.zeros(self.embed_dim))
#         #     self.register_buffer("z_log_var", torch.zeros(self.embed_dim))
#
#         self.register_buffer('embeddings', torch.randn(num_codebook, embed_dim))
#         self.register_buffer('N', torch.zeros(num_codebook))
#         self.register_buffer('z_avg', self.embeddings.data.clone())
#
#         # this is exact count and is different from self.codebook.vq_count.
#         self.register_buffer("vq_count", torch.zeros(self.num_codebook), persistent=False)
#         self.use_restart = use_restart
#         self.use_split = use_split
#         self.use_gumbel = use_gumbel
#
#         self.update_indices = None  # placeholder
#         self.update_candidates = None  # placeholder
#         self._initialized = False
#         self.jsd = JSDLoss(reduction="batchmean")
#         self.use_weighted_sum = use_weighted_sum
#         self.update_norm = update_norm
#         self.eps = eps
#
#     @torch.no_grad()
#     def prepare_restart(self, vq_current_count: torch.Tensor, z_flat: torch.Tensor) -> None:
#         """Restart dead entries
#         :param vq_current_count:        (K,)
#         :param z_flat:                  (n, d) = (bhw, d)
#         """
#         n_data = z_flat.shape[0]
#         update_indices = torch.nonzero(vq_current_count == 0, as_tuple=True)[0]
#         n_update = len(update_indices)
#
#         z_indices = list(range(n_data))
#         random.shuffle(z_indices)
#
#         if n_update <= n_data:
#             z_indices = z_indices[:n_update]
#         else:
#             update_indices = update_indices.tolist()
#             random.shuffle(update_indices)
#             update_indices = update_indices[:n_data]
#
#         self.update_indices = update_indices
#         self.update_candidates = z_flat[z_indices]
#
#     @torch.no_grad()
#     def restart(self) -> None:
#         if (self.update_indices is not None) and (self.update_candidates is not None):
#             self.embeddings.data[self.update_indices] = self.update_candidates
#             self.vq_count.fill_(0)
#             self.reset()
#
#             self.update_indices = None
#             self.update_candidates = None
#
#     @torch.no_grad()
#     def reset(self) -> None:
#         self.z_avg.data.copy_(self.embeddings.data)
#         self.vq_count.fill_(0)
#
#     @torch.no_grad()
#     def split(self, vq_current_count: torch.Tensor) -> int:
#         """Split replace dead entries
#         :param vq_current_count:        (K,)
#         :return:                        codebooks that are not used
#         """
#         update_indices = torch.nonzero(vq_current_count == 0, as_tuple=True)[0]
#         if len(update_indices) == 0:
#             return 0
#         n_update = len(update_indices)
#         update_indices = update_indices[torch.randperm(n_update)]
#
#         vq_total_count = self.vq_count  # (K,)
#         vq_weight = self.embeddings  # (K, d)
#         vq_weight_avg = self.z_avg  # (K, d)
#
#         _, vq_total_count_sort = torch.sort(vq_total_count, dim=0, descending=True)
#         vq_total_count_sort = vq_total_count_sort[:n_update]
#
#         vq_total_count_candidate = vq_total_count.detach().clone()
#         vq_weight_candidate = vq_weight.detach().clone()
#         vq_weight_avg_candidate = vq_weight_avg.detach().clone()
#
#         noise = torch.randn(n_update, self.embed_dim, dtype=vq_weight.dtype, device=vq_weight.device).mul_(0.02)
#         vq_weight_candidate[update_indices] = vq_weight[vq_total_count_sort] + noise
#         vq_weight_candidate[vq_total_count_sort] = vq_weight[vq_total_count_sort] - noise
#         vq_total_count_candidate[update_indices] = vq_total_count[vq_total_count_sort] / 2.0
#         vq_total_count_candidate[vq_total_count_sort] = vq_total_count[vq_total_count_sort] / 2.0
#         vq_weight_avg_candidate[update_indices] = vq_weight_avg[vq_total_count_sort] / 2.0
#         vq_weight_avg_candidate[vq_total_count_sort] = vq_weight_avg[vq_total_count_sort] / 2.0
#
#         self.vq_count.data.copy_(vq_total_count_candidate.data)
#         self.embeddings.data.copy_(vq_weight_candidate.data)
#         self.z_avg.data.copy_(vq_weight_avg_candidate.data)
#
#         self.vq_count.fill_(0)
#
#         return n_update
#
#     def _tile(self, x):
#         d, ew = x.shape  # (bhw, c)
#         std = 0.01 / np.sqrt(ew)
#         x = x + torch.randn_like(x) * std
#
#         return x
#
#     def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
#         """EMA-VQ forward
#         :param z:       (batch_size, embed_dim, h, w)
#         :return:        (batch_size, embed_dim, h, w) quantized
#                         loss = codebook_loss + beta * commitment_loss
#                         statistics
#         """
#         z = z.permute(0, 2, 3, 1).contiguous()
#         b, h, w, d = z.shape
#
#         z_flat = z.view(-1, d)  # (bhw, d) = (n, d)
#         n = b * h * w
#         k = self.num_codebook
#
#         if not self._initialized and self.training:
#             self._initialized = True
#             # y = self._tile(z_flat)
#             # _k_rand = y[torch.randperm(y.shape[0])][:self.num_codebook]
#             self.prepare_restart(torch.zeros(self.num_codebook, dtype=torch.long, device=z.device), z_flat)
#             _k_rand = self.update_candidates
#             broadcast_tensors(_k_rand, 0)
#             self.embeddings.data.copy_(_k_rand)
#             self.z_avg.data.copy_(_k_rand)
#             self.N.data.copy_(torch.ones(self.num_codebook))
#
#         # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
#         codebook = self.embeddings  # (K, d)
#
#         distance = (
#                 torch.sum(z_flat ** 2, dim=1, keepdim=True) +  # (n, d) -> (n, 1)
#                 torch.sum(codebook ** 2, dim=1) -  # (K, d) -> (K,) == (1, K)
#                 2 * torch.matmul(z_flat, codebook.t())  # (n, K)
#         )  # (n, K)
#
#         encoding_indices = torch.argmin(distance, dim=1)
#         encode_onehot = F.one_hot(encoding_indices, self.num_codebook).to(z_flat.dtype)
#         z_quantized = F.embedding(encoding_indices, self.embeddings)
#
#         distance_prob = F.softmax(-distance * 1.0, dim=1)  # (n, K) # TODO scaling
#
#         output = dict()
#
#         if self.training:
#             n_total = encode_onehot.sum(dim=0)  # (k)
#             encode_sum = z_flat.t() @ encode_onehot  # (d, n) * (n, k) -> (d, k)
#             n_total = all_reduce_tensor(n_total, op="sum")
#             encode_sum = all_reduce_tensor(encode_sum, op="sum")
#
#             self.N.data.mul_(self.decay).add_(n_total, alpha=1 - self.decay)
#             self.z_avg.data.mul_(self.decay).add_(encode_sum.t(), alpha=1 - self.decay)
#
#             n = self.N.sum()
#             weights = (self.N + self.eps) / (n + self.num_codebook * self.eps) * n
#             encode_normalized = self.z_avg / weights.unsqueeze(1)
#             self.embeddings.data.copy_(encode_normalized)
#
#             # y = self._tile(z_flat)
#             # _k_rand = y[torch.randperm(y.shape[0])][:self.num_codebook]
#             self.prepare_restart(torch.zeros(self.num_codebook, dtype=torch.long, device=z.device), z_flat)
#             _k_rand = self.update_candidates
#             broadcast_tensors(_k_rand, 0)
#
#             usage = (self.N.view(self.num_codebook, 1) >= 1).float()
#             self.embeddings.data.mul_(usage).add_(_k_rand * (1 - usage))  # 0 usage is randomly restarted.
#
#             self.vq_count += n_total
#
#             vq_current_hist = get_histogram_count(n_total, prefix="current")
#             vq_hist = get_histogram_count(self.vq_count, prefix="total")
#             output.update(vq_hist)
#             output.update(vq_current_hist)
#
#             if self.use_split:
#                 n_split = self.split(n_total)  # not-used count
#             else:
#                 n_split = len(torch.nonzero(n_total == 0, as_tuple=True)[0])
#             output["codebook-usage"] = (self.num_codebook - n_split) / self.num_codebook  # used ratio
#
#         # compute loss for embedding
#         commitment_loss = F.mse_loss(z_flat, z_quantized.detach())  # make input to be similar to codebook
#
#         loss = self.beta * commitment_loss
#         output["loss"] = loss
#         output["commitment-loss"] = commitment_loss
#         output["codebook-sum"] = torch.sum(torch.abs(self.embeddings.data))
#
#         if not self.use_weighted_sum:
#             z_quantized = z_flat + (z_quantized - z_flat).detach()  # (n, d)
#
#         # reshape back to match original input shape
#         q = z_quantized.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()
#
#         return q, output, distance_prob
#
#     def extra_repr(self) -> str:
#         return f"embed_dim={self.embed_dim}, " \
#                f"num_codebook={self.num_codebook}, " \
#                f"normalize={self.normalize}, " \
#                f"use_gumbel={self.use_gumbel}, " \
#                f"use_split={self.use_split}, " \
#                f"use_weighted_sum={self.use_weighted_sum}, " \
#                f"use_restart={self.use_restart}"


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
                 quantizer_cls=EMAVectorQuantizer,
                 ) -> None:
        super().__init__()

        if embed_dim % num_pq != 0:
            raise ValueError(f"Embed dim {embed_dim} should be divisible by #PQ {num_pq}.")
        self.num_pq = num_pq
        self.pq_dim = embed_dim // num_pq

        self.quantizers = nn.ModuleList([
            quantizer_cls(num_codebook, self.pq_dim, beta=beta, normalize=normalize,
                          decay=decay, eps=eps,
                          use_restart=use_restart, use_gumbel=use_gumbel, use_split=use_split,
                          use_weighted_sum=use_weighted_sum, update_norm=update_norm)
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

    def extra_repr(self) -> str:
        return f"num_pq={self.num_pq}"
