import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class InfoNCELoss(nn.Module):
    def __init__(self,
                 normalize: str = "l2",
                 temperature: float = 1.0,
                 reduction: str = "mean"):
        super().__init__()

        self.normalize = normalize
        self.temperature = temperature
        self.reduction = reduction
        self.num_neg = 1000

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        :param x1: (b, d, h, w) -> (b, h, w, d) -> (bhw, d)
        :param x2: (b, d, h, w) -> (b, h, w, d) -> (bhw, d)
               neg: (bhw, k, d), k = neg_sample per pixel

        :return:
        """
        b, d, h, w = x1.shape
        x1 = x1.permute(0, 2, 3, 1).contiguous()  # (b, h, w, d)
        flat_x1 = x1.view(-1, d)  # (bhw, d)

        x2 = x2.permute(0, 2, 3, 1).contiguous()
        flat_x2 = x2.view(-1, d)

        # TODO except self-sample + 4-part
        # make paired random negative sample
        rand_neg = torch.zeros(b * h * w, self.num_neg, d, device=x1.device)  # (bhw, n, d)
        indices = torch.randint(b * h * w, (b * h * w, self.num_neg), device=x1.device)
        for idx, data in enumerate(indices):
            rand_neg[idx] = flat_x1[data]
        # rand_indices = list(range((b-1) * h * w))
        # random.shuffle(rand_indices)
        # indices = rand_indices[:self.num_neg]
        # rand_neg = flat_x1[indices]

        if self.normalize == "l2":
            x1_norm = F.normalize(flat_x1, dim=-1)
            x2_norm = F.normalize(flat_x2, dim=-1)
            neg_norm = F.normalize(rand_neg, dim=-1)
        # elif self.normalize == "z_norm":  # z-normalize
        #     x1_flat_std, x1_flat_mean = torch.std_mean(flat_x1, dim=1, keepdim=True)  # (n, 1)
        #     x1_norm = (flat_x1 - x1_flat_mean) / (x1_flat_std + 1e-5)
        #
        #     x2_flat_std, x2_flat_mean = torch.std_mean(flat_x2, dim=1, keepdim=True)  # (n, 1)
        #     x2_norm = (flat_x2 - x2_flat_mean) / (x2_flat_std + 1e-5)
        # elif self.normalize == "none":
        #     x1_norm = x1
        #     x2_norm = x2
        else:
            raise ValueError(f"Unsupported normalize type {self.normalize}")

        pos_similarity = torch.matmul(x1_norm, x2_norm.T)  # (bhw, d) * (d, bhw) -> (bhw, bhw)
        pos_similarity = torch.exp(pos_similarity / self.temperature)
        pos = torch.mean(pos_similarity, dim=1)

        x1_norm = x1_norm.unsqueeze(1)
        neg_similarity = torch.matmul(x1_norm, neg_norm.transpose(-2, -1))  # (bhw, 1, d) * (bhw, d, k) -> (bhw, 1, k)
        neg_similarity = neg_similarity.squeeze(1)  # (bhw, k)
        neg_similarity = torch.exp(neg_similarity / self.temperature)
        neg = torch.mean(neg_similarity, dim=1)

        loss = - (torch.log(pos) - torch.log(pos + neg))

        if self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "mean":
            loss = torch.mean(loss)

        return loss


class CLUBLoss(nn.Module):
    def __init__(self,
                 reduction: str = "mean"):
        super().__init__()

        self.reduction = reduction

    def forward(self,
                x: torch.Tensor,
                mu: torch.Tensor,
                logvar: torch.Tensor) -> torch.Tensor:
        '''

        :param x:  (b, d, h, w)
        :param mu:
        :param logvar:
        :return:
        '''
        x = x.permute(0, 2, 3, 1).contiguous()
        b, h, w, d = x.shape
        flat_x = x.view(-1, d)

        pos = -0.5 * torch.sum(
            torch.square(flat_x - mu) / torch.exp(logvar), dim=-1
        )

        neg = -0.5 * torch.mean(
            torch.sum(
                torch.square(flat_x.unsqueeze(0) - mu.unsqueeze(1)) /
                torch.exp(logvar.unsqueeze(1)),
                dim=-1
            ),
            dim=-1
        )
        loss = pos - neg

        if self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "mean":
            loss = torch.mean(loss)

        return loss

    def cal_distance(self, x, y):
        distance = (
                torch.sum(x ** 2, dim=1, keepdim=True) +  # (bhw, d) -> (bhw, 1)
                torch.sum(y ** 2, dim=1) -  # (K, d) -> (K,) == (1, K)
                2 * torch.matmul(x, y.t())  # (n, K)
        )

        return distance


class JSDLoss(nn.Module):
    def __init__(self, reduction="batchmean"):
        super().__init__()
        self.kl = nn.KLDivLoss(reduction=reduction, log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        """
        :param p:    (bhw, K)
        :param q:    (bhw, K)
        :return:
        """
        if torch.min(p) < 0.0 or torch.max(p) > 1.0 or torch.min(q) < 0 or torch.max(q) > 1.0:
            raise ValueError(
                f"min_p, max_p, min_q, max_q : {torch.min(p)}, {torch.max(p)}, {torch.min(q)}, {torch.max(q)}")

        m = (0.5 * (p + q).add(1e-6)).log()
        # TODO check position
        return 0.5 * (self.kl(m, p.add(1e-6).log()) + self.kl(m, q.add(1e-6).log()))


def tensor_correlation(a, b):
    return torch.einsum("nchw,ncij->nhwij", a, b)


def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)


def sample(t: torch.Tensor, coords: torch.Tensor):
    return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)


@torch.jit.script
def super_perm(size: int, device: torch.device):
    perm = torch.randperm(size, device=device, dtype=torch.long)
    perm[perm == torch.arange(size, device=device)] += 1
    return perm % size


class ContrastiveCorrelationLoss(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

    def standard_scale(self, t):
        t1 = t - t.mean()
        t2 = t1 / t1.std()
        return t2

    def helper(self, f1, f2, c1, c2, shift):
        with torch.no_grad():
            # Comes straight from backbone which is currently frozen. this saves mem.
            fd = tensor_correlation(norm(f1), norm(f2))

            if self.cfg["pointwise"]:
                old_mean = fd.mean()
                fd -= fd.mean([3, 4], keepdim=True)
                fd = fd - fd.mean() + old_mean

        cd = tensor_correlation(norm(c1), norm(c2))

        if self.cfg["zero_clamp"]:
            min_val = 0.0
        else:
            min_val = -9999.0

        if self.cfg["stabilize"]:
            loss = - cd.clamp(min_val, .8) * (fd - shift)
        else:
            loss = - cd.clamp(min_val) * (fd - shift)

        return loss, cd

    def forward(self,
                orig_feats: torch.Tensor,
                orig_feats_pos: torch.Tensor,
                orig_code: torch.Tensor,
                orig_code_pos: torch.Tensor,
                ):
        coord_shape = [orig_feats.shape[0], 11, 11, 2]

        coords1 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
        coords2 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1

        feats = sample(orig_feats, coords1)
        code = sample(orig_code, coords1)

        feats_pos = sample(orig_feats_pos, coords2)
        code_pos = sample(orig_code_pos, coords2)
        pos_intra_loss, pos_intra_cd = self.helper(
            feats, feats, code, code, self.cfg["pos_intra_shift"])
        pos_inter_loss, pos_inter_cd = self.helper(
            feats, feats_pos, code, code_pos, self.cfg["pos_inter_shift"])

        neg_losses = []
        neg_cds = []
        for i in range(self.cfg["neg_samples"]):
            perm_neg = super_perm(orig_feats.shape[0], orig_feats.device)
            feats_neg = sample(orig_feats[perm_neg], coords2)
            code_neg = sample(orig_code[perm_neg], coords2)
            neg_inter_loss, neg_inter_cd = self.helper(
                feats, feats_neg, code, code_neg, self.cfg["neg_inter_shift"])
            neg_losses.append(neg_inter_loss)
            neg_cds.append(neg_inter_cd)

        neg_inter_loss = torch.cat(neg_losses, axis=0)
        neg_inter_cd = torch.cat(neg_cds, axis=0)

        return (self.cfg["pos_intra_weight"] * pos_intra_loss.mean() +
                self.cfg["pos_inter_weight"] * pos_inter_loss.mean() +
                self.cfg["neg_inter_weight"] * neg_inter_loss.mean(),
                pos_intra_loss.mean().item(),
                pos_inter_loss.mean().item(),
                neg_inter_loss.mean().item()
                )
