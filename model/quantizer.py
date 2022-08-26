"""https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py"""
import random
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from model.loss import JSDLoss
from utils.dist_utils import all_reduce_tensor

__all__ = ["VectorQuantizer", "EMAVectorQuantizer", "EmbeddingEMA"]


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
                 ) -> None:
        super().__init__()
        self.num_codebook = num_codebook
        self.embed_dim = embed_dim
        self.beta = beta
        self.normalize = normalize
        if normalize == "z_trainable":
            self.z_mean = nn.Parameter(torch.zeros(self.embed_dim))
            self.z_log_var = nn.Parameter(torch.zeros(self.embed_dim))

        self.codebook = nn.Embedding(self.num_codebook, self.embed_dim)  # codebook: (K, d)
        # self.codebook.weight.data.uniform_(-1.0 / self.num_codebook, 1.0 / self.num_codebook)  # TODO initialize diff?
        nn.init.xavier_normal_(self.codebook.weight)

        self.register_buffer("vq_count", torch.zeros(self.num_codebook), persistent=False)
        self.use_restart = use_restart
        self.use_gumbel = use_gumbel
        self.use_split = use_split
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
               f"use_restart={self.use_restart}"


class EmbeddingEMA(nn.Module):
    def __init__(self,
                 num_codebook: int,
                 embed_dim: int,
                 decay: float = 0.99,
                 eps: float = 1e-5
                 ) -> None:
        super().__init__()
        self.decay = decay
        self.eps = eps
        self.num_codebook = num_codebook

        weight = torch.randn(num_codebook, embed_dim)
        nn.init.uniform_(weight, -1.0 / num_codebook, 1.0 / num_codebook)
        # nn.init.xavier_uniform_(weight)

        self.register_buffer("weight", weight)
        self.register_buffer("weight_avg", weight.clone())
        self.register_buffer("vq_count", torch.zeros(num_codebook))

    @torch.no_grad()
    def reset(self) -> None:
        self.weight_avg.data.copy_(self.weight.data)
        self.vq_count.fill_(0)

    def forward(self, indices: torch.Tensor, is_sum: bool = False) -> torch.Tensor:
        if is_sum:
            embeded = torch.matmul(indices, self.weight)
        else:
            embeded = F.embedding(indices, self.weight)
        return embeded

    def update(self, vq_current_count, vq_current_sum) -> None:
        # EMA count update
        self.vq_count_ema_update(vq_current_count)
        # EMA weight average update
        self.weight_avg_ema_update(vq_current_sum)
        # normalize embed_avg and update weight
        self.weight_update()

    def vq_count_ema_update(self, vq_current_count) -> None:
        self.vq_count.data.mul_(self.decay).add_(vq_current_count, alpha=1 - self.decay)

    def weight_avg_ema_update(self, vq_current_sum) -> None:
        self.weight_avg.data.mul_(self.decay).add_(vq_current_sum, alpha=1 - self.decay)

    def weight_update(self) -> None:
        n = self.vq_count.sum()
        smoothed_cluster_size = (
                (self.vq_count + self.eps) / (n + self.num_codebook * self.eps) * n
        )  # Laplace smoothing
        # normalize embedding average with smoothed cluster size
        weight_normalized = self.weight_avg / smoothed_cluster_size.unsqueeze(1)  # (unsqueeze channel)
        self.weight.data.copy_(weight_normalized)


class EMAVectorQuantizer(nn.Module):
    def __init__(self,
                 num_codebook: int,
                 embed_dim: int,
                 beta: float = 0.25,  # commitment loss
                 normalize: Optional[str] = None,
                 decay: float = 0.99,
                 eps: float = 1e-5,
                 use_restart: bool = False,
                 use_gumbel: bool = False,
                 use_split: bool = False,
                 ) -> None:
        super().__init__()
        self.num_codebook = num_codebook
        self.embed_dim = embed_dim
        self.beta = beta
        self.decay = decay
        self.normalize = normalize
        if normalize == "z_trainable":
            self.register_buffer("z_mean", torch.zeros(self.embed_dim))
            self.register_buffer("z_log_var", torch.zeros(self.embed_dim))

        self.codebook = EmbeddingEMA(self.num_codebook, self.embed_dim,
                                     decay=decay, eps=eps)  # codebook: (K, d)

        # this is exact count and is different from self.codebook.vq_count.
        self.register_buffer("vq_count", torch.zeros(self.num_codebook), persistent=False)
        self.use_restart = use_restart
        self.use_split = use_split
        self.use_gumbel = use_gumbel

        self.update_indices = None  # placeholder
        self.update_candidates = None  # placeholder
        # self._initialized = False
        self.jsd = JSDLoss(reduction="batchmean")

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
            self.codebook.weight.data[self.update_indices] = self.update_candidates
            self.vq_count.fill_(0)
            self.codebook.reset()

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

        vq_total_count = self.codebook.vq_count  # (K,)
        vq_weight = self.codebook.weight  # (K, d)
        vq_weight_avg = self.codebook.weight_avg  # (K, d)

        _, vq_total_count_sort = torch.sort(vq_total_count, dim=0, descending=True)
        vq_total_count_sort = vq_total_count_sort[:n_update]

        vq_total_count_candidate = vq_total_count.detach().clone()
        vq_weight_candidate = vq_weight.detach().clone()
        vq_weight_avg_candidate = vq_weight_avg.detach().clone()

        noise = torch.randn(n_update, self.embed_dim, dtype=vq_weight.dtype, device=vq_weight.device).mul_(0.02)
        vq_weight_candidate[update_indices] = vq_weight[vq_total_count_sort] + noise
        vq_weight_candidate[vq_total_count_sort] = vq_weight[vq_total_count_sort] - noise
        vq_total_count_candidate[update_indices] = vq_total_count[vq_total_count_sort] / 2.0
        vq_total_count_candidate[vq_total_count_sort] = vq_total_count[vq_total_count_sort] / 2.0
        vq_weight_avg_candidate[update_indices] = vq_weight_avg[vq_total_count_sort] / 2.0
        vq_weight_avg_candidate[vq_total_count_sort] = vq_weight_avg[vq_total_count_sort] / 2.0

        # for i in range(len(update_indices)):
        #     noise = torch.randn(self.embed_dim, dtype=vq_weight.dtype, device=vq_weight.device).mul_(0.02)
        #     vq_code = vq_weight[vq_total_count_sort[i]]
        #     vq_code_pos = vq_code + noise
        #     vq_code_neg = vq_code - noise
        #     vq_code_count = vq_total_count[vq_total_count_sort[i]] / 2.0
        #     vq_code_avg = vq_weight_avg[vq_total_count_sort[i]] / 2.0
        #
        #     vq_weight_candidate[update_indices[i]] = vq_code_pos
        #     vq_weight_candidate[vq_total_count_sort[i]] = vq_code_neg
        #     vq_total_count_candidate[update_indices[i]] = vq_code_count
        #     vq_total_count_candidate[vq_total_count_sort[i]] = vq_code_count
        #     vq_weight_avg_candidate[update_indices[i]] = vq_code_avg
        #     vq_weight_avg_candidate[vq_total_count_sort[i]] = vq_code_avg

        self.codebook.vq_count.data.copy_(vq_total_count_candidate.data)
        self.codebook.weight.data.copy_(vq_weight_candidate.data)
        self.codebook.weight_avg.data.copy_(vq_weight_avg_candidate.data)

        self.vq_count.fill_(0)
        return n_update

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """EMA-VQ forward
        :param z:       (batch_size, embed_dim, h, w)
        :return:        (batch_size, embed_dim, h, w) quantized
                        loss = codebook_loss + beta * commitment_loss
                        statistics
        """
        z = z.permute(0, 2, 3, 1).contiguous()
        b, h, w, d = z.shape

        z_flat = z.view(-1, d)  # (bhw, d) = (n, d)
        n = b * h * w
        k = self.num_codebook

        # if not self._initialized:
        #     self.prepare_restart(torch.zeros(self.num_codebook, dtype=torch.long, device=z.device),
        #                          z_flat)
        #     self.restart()
        #     self._initialized = True

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

        distance = (
                torch.sum(z_norm ** 2, dim=1, keepdim=True) +  # (n, d) -> (n, 1)
                torch.sum(codebook_norm ** 2, dim=1) -  # (K, d) -> (K,) == (1, K)
                2 * torch.matmul(z_norm, codebook_norm.t())  # (n, K)
        )  # (n, K)

        if self.training and self.use_gumbel:
            distance_prob_gumbel = F.gumbel_softmax(-distance / 0.01, tau=1.0, hard=True, dim=1)
            vq_indices = torch.argmax(distance_prob_gumbel, dim=1)
        else:
            vq_indices = torch.argmin(distance, dim=1)  # (n,) : index of the closest code vector.
        distance_prob = F.softmax(-distance, dim=1)  # (n, K)
        # z_quantized = self.codebook(vq_indices)  # (n, d)
        z_norm_quantized = F.embedding(vq_indices, codebook_norm)  # (n, d)
        # z_norm_quantized = torch.matmul(distance_prob, codebook_norm)  # (n, d)

        output = dict()

        if self.training:
            with torch.no_grad():
                vq_indices_one_hot = F.one_hot(vq_indices, self.num_codebook).to(z.dtype)  # (n, K)
                vq_current_count = torch.sum(vq_indices_one_hot, dim=0)  # (K,)
                vq_current_sum = torch.matmul(vq_indices_one_hot.t(), z_flat)  # (K, n) x (n, d) = (K, d)

                vq_current_count = all_reduce_tensor(vq_current_count, op="sum")
                vq_current_sum = all_reduce_tensor(vq_current_sum, op="sum")

                self.vq_count += vq_current_count

                vq_current_hist = get_histogram_count(vq_current_count, prefix="current")
                vq_hist = get_histogram_count(self.vq_count, prefix="total")
                output.update(vq_hist)
                output.update(vq_current_hist)

                if self.use_restart:
                    self.prepare_restart(vq_current_count, z_flat)

                # codebook update
                self.codebook.update(vq_current_count, vq_current_sum)

                if self.use_split:
                    n_split = self.split(vq_current_count)  # not-used count
                else:
                    n_split = len(torch.nonzero(vq_current_count == 0, as_tuple=True)[0])
                output["codebook-usage"] = (self.num_codebook - n_split) / self.num_codebook  # used ratio

        # compute loss for embedding
        commitment_loss = F.mse_loss(z_norm, z_norm_quantized.detach())  # make input to be similar to codebook

        # jsd loss
        # distance_prob_2d = distance_prob.view(b, h, w, k)  # (n, K) -> (b, h, w, K)
        # jsd_h = self.jsd(
        #     distance_prob_2d[:, 1:, :, :].contiguous().view(-1, k),
        #     distance_prob_2d[:, :-1, :, :].contiguous().view(-1, k),
        # )
        # jsd_w = self.jsd(
        #     distance_prob_2d[:, :, 1:, :].contiguous().view(-1, k),
        #     distance_prob_2d[:, :, :-1, :].contiguous().view(-1, k),
        # )
        # jsd_loss = (jsd_h + jsd_w) / 2.0

        loss = self.beta * commitment_loss
        # loss += (self.beta * 1.0) * jsd_loss  # TODO heuristic

        output["loss"] = loss
        output["commitment-loss"] = commitment_loss
        # output["jsd-loss"] = jsd_loss
        output["codebook-sum"] = torch.sum(torch.abs(self.codebook.weight))

        # preserve gradients
        # z_quantized = z_flat + (z_quantized - z_flat).detach()  # (n, d)
        z_norm_quantized = z_norm + (z_norm_quantized - z_norm).detach()  # (n, d)

        # reshape back to match original input shape
        # q = z_quantized.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()
        q = z_norm_quantized.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()

        return q, output, distance_prob

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, " \
               f"num_codebook={self.num_codebook}, " \
               f"normalize={self.normalize}, " \
               f"use_gumbel={self.use_gumbel}, " \
               f"use_split={self.use_split}, " \
               f"use_restart={self.use_restart}"
