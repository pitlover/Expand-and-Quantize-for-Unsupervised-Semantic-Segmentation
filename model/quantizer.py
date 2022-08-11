"""https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from utils.dist_utils import all_reduce_tensor

__all__ = ["VectorQuantizer", "EMAVectorQuantizer"]


@torch.no_grad()
def get_histogram_count(count: torch.Tensor, prefix: str = "") -> Dict:
    prob = count.float() / count.sum()  # (K,)
    prob, _ = torch.sort(prob, dim=0, descending=True)  # (K,)
    c_sum = torch.cumsum(prob, dim=0)  # (K,)
    output = {f"{prefix}-p10": None, f"{prefix}-p50": None, f"{prefix}-p90": None}
    for i in range(len(c_sum)):
        if (c_sum[i] >= 0.9) and (output[f"{prefix}-p90"] is None):
            output[f"{prefix}-p90"] = i
        if (c_sum[i] >= 0.5) and (output[f"{prefix}-p50"] is None):
            output[f"{prefix}-p50"] = i
        if (c_sum[i] >= 0.1) and (output[f"{prefix}-p10"] is None):
            output[f"{prefix}-p10"] = i
    return output


class VectorQuantizer(nn.Module):

    def __init__(self,
                 num_codebook: int,
                 embed_dim: int,
                 beta: float = 0.25,  # commitment loss
                 normalize: Optional[str] = None
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
        self.codebook.weight.data.uniform_(-1.0 / self.num_codebook, 1.0 / self.num_codebook)  # [-1/K, 1/K]

        self.vq_count = torch.ones(self.num_codebook, dtype=torch.long)  # (K,)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
            z_flat = F.normalize(z_flat, dim=1)
            codebook = F.normalize(codebook, dim=1)
        elif self.normalize == "z_norm":  # z-normalize
            z_flat_std, z_flat_mean = torch.std_mean(z_flat, dim=1, keepdim=True)  # (n, 1)
            z_flat = (z_flat - z_flat_mean) / (z_flat_std + 1e-5)

            codebook_std, codebook_mean = torch.std_mean(codebook, dim=1, keepdim=True)  # (K, 1)
            codebook = (codebook - codebook_mean) / codebook_std
        elif self.normalize == "z_trainable":
            z_flat_mean = self.z_mean
            z_flat_std = self.z_log_var.exp().sqrt()
            z_flat = (z_flat - z_flat_mean) / (z_flat_std + 1e-5)
        elif self.normalize == "none":
            pass
        else:
            raise ValueError(f"Unsupported normalize type {self.normalize}")

        distance = (
                torch.sum(z_flat ** 2, dim=1, keepdim=True) +  # (n, d) -> (n, 1)
                torch.sum(codebook ** 2, dim=1) -  # (K, d) -> (K,) == (1, K)
                2 * torch.matmul(z_flat, codebook.t())  # (n, K)
        )  # (n, K)

        vq_indices = torch.argmin(distance, dim=1)  # (n,) : index of the closest code vector.
        z_quantized = self.codebook(vq_indices)  # (n, d)

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

            vq_count_top10 = torch.topk(self.vq_count, k=10, largest=True).values
            vq_count_bottom10 = torch.topk(self.vq_count, k=10, largest=False).values
            output["top10"] = vq_count_top10
            output["bot10"] = vq_count_bottom10

        # compute loss for embedding
        codebook_loss = F.mse_loss(z_quantized, z_flat.detach())  # make codebook to be similar to input
        commitment_loss = F.mse_loss(z_flat, z_quantized.detach())  # make input to be similar to codebook
        loss = codebook_loss + self.beta * commitment_loss

        output["loss"] = loss
        output["codebook_loss"] = codebook_loss
        output["commitment_loss"] = commitment_loss

        # preserve gradients
        z_quantized = z_flat + (z_quantized - z_flat).detach()  # (n, d)

        # reshape back to match original input shape
        q = z_quantized.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()

        return q, output


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

        self.register_buffer("weight", weight)
        self.register_buffer("weight_avg", weight.clone())
        self.register_buffer("vq_count", torch.ones(num_codebook))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.weight)

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
        weight_normalized = self.weight_avg / smoothed_cluster_size.unsqueeze(1)
        self.weight.data.copy_(weight_normalized)


class EMAVectorQuantizer(nn.Module):
    def __init__(self,
                 num_codebook: int,
                 embed_dim: int,
                 beta: float = 0.25,  # commitment loss
                 normalize: Optional[str] = None,
                 decay: float = 0.99,
                 eps: float = 1e-5,
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
        self.vq_count = torch.ones(self.num_codebook, dtype=torch.float32)  # (K,)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """EMA-VQ forward
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
            z_flat = F.normalize(z_flat, dim=1)
            codebook = F.normalize(codebook, dim=1)
        elif self.normalize == "z_norm":  # z-normalize
            z_flat_std, z_flat_mean = torch.std_mean(z_flat, dim=1, keepdim=True)  # (n, 1)
            z_flat = (z_flat - z_flat_mean) / (z_flat_std + 1e-5)

            codebook_std, codebook_mean = torch.std_mean(codebook, dim=1, keepdim=True)  # (K, 1)
            codebook = (codebook - codebook_mean) / codebook_std
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

            z_flat = (z_flat - z_flat_mean) / (z_flat_std + 1e-5)
        elif self.normalize == "none":
            pass
        else:
            raise ValueError(f"Unsupported normalize type {self.normalize}")

        distance = (
                torch.sum(z_flat ** 2, dim=1, keepdim=True) +  # (n, d) -> (n, 1)
                torch.sum(codebook ** 2, dim=1) -  # (K, d) -> (K,) == (1, K)
                2 * torch.matmul(z_flat, codebook.t())  # (n, K)
        )  # (n, K)

        vq_indices = torch.argmin(distance, dim=1)  # (n,) : index of the closest code vector.
        z_quantized = self.codebook(vq_indices)  # (n, d)

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

                vq_count_top10 = torch.topk(self.vq_count, k=10, largest=True).values
                vq_count_bottom10 = torch.topk(self.vq_count, k=10, largest=False).values
                output["top10"] = vq_count_top10
                output["bot10"] = vq_count_bottom10

                # codebook update
                self.codebook.update(vq_current_count, vq_current_sum)

        # compute loss for embedding
        commitment_loss = F.mse_loss(z_flat, z_quantized.detach())  # make input to be similar to codebook
        loss = self.beta * commitment_loss

        output["loss"] = loss
        output["commitment_loss"] = commitment_loss

        # preserve gradients
        z_quantized = z_flat + (z_quantized - z_flat).detach()  # (n, d)

        # reshape back to match original input shape
        q = z_quantized.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()

        return q, output
