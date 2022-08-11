from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from utils.crf_utils import batched_crf

__all__ = ["UnSegEvaluator"]


class UnSegEvaluator(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_classes: int,
                 extra_classes: int = 0
                 ) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.linear_probe = nn.Conv2d(embed_dim, num_classes, kernel_size=1, stride=1)
        self.cluster_probe = ClusterLookup(embed_dim, num_classes + extra_classes)

    def forward_linear(self, feat: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        :param feat:        (batch_size, 384, 28, 28)
        :param label:       (batch_size, 224, 224)
        :return:            loss
        """
        logit = self.linear_probe(feat)
        logit = F.interpolate(logit, label.shape[-2:], mode="bilinear", align_corners=True)  # maybe False?

        label_flat = label.view(-1)
        logit_flat = logit.permute(0, 2, 3, 1).reshape(-1, self.num_classes)

        mask = torch.logical_and(label_flat >= 0, label_flat < self.num_classes)
        label_flat = label_flat[mask]
        logit_flat = logit_flat[mask]
        loss = F.cross_entropy(logit_flat, label_flat, reduction="mean")
        return loss

    def forward(self,
                out: torch.Tensor,
                img: torch.Tensor,
                label: Optional[torch.Tensor] = None,
                is_crf: bool = False
                ) -> Tuple[torch.Tensor, ...]:

        if out.shape[-2:] != img.shape[-2:]:
            out = F.interpolate(out, img.shape[-2:], mode="bilinear", align_corners=True)  # maybe False?

        if is_crf:
            linear_log_prob = torch.log_softmax(self.linear_probe(out), dim=1)
            cluster_loss, cluster_log_prob = self.cluster_probe(out, 2, log_probs=True)

            linear_preds = batched_crf(img, linear_log_prob).argmax(1)
            cluster_preds = batched_crf(img, cluster_log_prob).argmax(1)

            linear_loss = torch.zeros_like(cluster_loss)  # 0
        else:
            assert label is not None
            linear_logits = self.linear_probe(out)
            linear_preds = linear_logits.argmax(1)
            cluster_loss, cluster_preds = self.cluster_probe(out, None)
            cluster_preds = cluster_preds.argmax(1)

            label_flat = label.view(-1)
            logit_flat = linear_logits.permute(0, 2, 3, 1).reshape(-1, self.num_classes)

            mask = torch.logical_and(label_flat >= 0, label_flat < self.num_classes)
            label_flat = label_flat[mask]
            logit_flat = logit_flat[mask]
            linear_loss = F.cross_entropy(logit_flat, label_flat, reduction="mean")

        return linear_loss, linear_preds, cluster_loss, cluster_preds


class ClusterLookup(nn.Module):

    def __init__(self, dim: int, n_classes: int):
        super(ClusterLookup, self).__init__()
        self.n_classes = n_classes
        self.dim = dim
        self.clusters = torch.nn.Parameter(torch.randn(n_classes, dim))

    def forward(self, x: torch.Tensor, alpha: Optional[float] = 2.0, log_probs: bool = False):  # feats, code

        normed_clusters = F.normalize(self.clusters, dim=1)  # (n_class, dim)
        normed_features = F.normalize(x, dim=1)  # (b, dim, h, w)

        inner_products = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)  # (b, n_class, h, w) # noqa

        if alpha is None:
            cluster_probs = F.one_hot(torch.argmax(inner_products, dim=1), self.n_classes)  # (b, h, w, n_class)
            cluster_probs = cluster_probs.permute(0, 3, 1, 2).contiguous().to(torch.float32)
        else:
            cluster_probs = F.softmax(inner_products * alpha, dim=1)  # alpha = 1 / temperature

        cluster_loss = -torch.sum(cluster_probs * inner_products, dim=1).mean()

        if log_probs:
            return cluster_loss, F.log_softmax(inner_products * alpha, dim=1)
        else:
            return cluster_loss, cluster_probs
