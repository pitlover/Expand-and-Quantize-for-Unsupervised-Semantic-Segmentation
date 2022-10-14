from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import pairwise_cosine_similarity
from utils.dist_utils import all_reduce_tensor


def distance(x: torch.Tensor, b: torch.Tensor, num_neg: int) -> torch.Tensor:
    '''

    :param x: flat_x => (bhw, d)
    :param b: batch_size
    :return:
    '''
    jump = x.shape[0] // b  # hw
    split_x = torch.chunk(x, chunks=b, dim=0)  # (bhw/b, d)
    neg = torch.zeros(x.shape[0], num_neg, x.shape[1], device=x.device)

    for iter in range(b):
        distance_ = torch.cdist(split_x[iter], x)  # (bhw/b, bhw)
        negative_index = torch.topk(distance_, num_neg, dim=-1)  # (bhw/b, n)
        neg_ = F.embedding(negative_index.indices, x)  # ( bhw/b, n, d)
        neg[iter * jump: (iter + 1) * jump] = neg_
        del distance_, negative_index, neg_
    del split_x
    return neg


class InfoNCELoss(nn.Module):
    def __init__(self,
                 normalize: str = "l2",
                 temperature: float = 1.0,
                 neg_sample: int = 100,
                 reduction: str = "mean",
                 cal_type: str = "random"):
        super().__init__()

        self.normalize = normalize
        self.temperature = temperature
        self.reduction = reduction
        self.num_neg = neg_sample
        self.cal_type = cal_type

    def _normalize(self, x):
        if self.normalize == "l2":
            x_norm = F.normalize(x, dim=-1)
        elif self.normalize == "z_norm":  # z-normalize
            x_std, x_mean = torch.std_mean(x, dim=1, keepdim=True)  # (n, 1)
            x_norm = (x - x_mean) / (x_std + 1e-5)
        elif self.normalize == "none":
            x_norm = x
        else:
            raise ValueError(f"Unsupported normalize type {self.normalize}")

        return x_norm

    def cosine(self, x: torch.Tensor, num_neg: int) -> torch.Tensor:
        '''

        :param x: (bhw, d) -> normalized_flat_x
        :param b: batch_size
        :param num_neg: number of negative samples
        :return:
        '''
        # jump = x.shape[0] // b  # hw
        # split_x = torch.chunk(x, chunks=b, dim=0)  # (bhw/b, d)
        # neg = torch.zeros(x.shape[0], num_neg, x.shape[1], device=x.device)  # (bhw, n, d)
        #
        # for iter in range(b):
        #     cos_similarity_ = pairwise_cosine_similarity(split_x[iter], x)  # (bhw/b, bhw) high value -> high same
        #     negative_index = torch.topk(cos_similarity_, num_neg, largest=False, dim=-1)  # (bhw/b, n) small similarity
        #     neg_ = F.embedding(negative_index.indices, x)  # (bhw/b, n, d)
        #     neg[iter * jump: (iter + 1) * jump] = neg_
        #     del cos_similarity_, negative_index, neg_
        # del split_x
        cos_similarity = pairwise_cosine_similarity(x)
        negative_index = torch.topk(cos_similarity, num_neg, largest=False, dim=-1)  # (bhw/b, n) small similarity
        neg = F.embedding(negative_index.indices, x)
        del cos_similarity
        return neg

    def random(self, x) -> torch.Tensor:  # TODO exclude myself
        bhw, d = x.shape
        rand_neg = torch.zeros(bhw, self.num_neg, d, device=x.device)  # (bhw, n, d)
        indices = torch.randint(bhw, (bhw, self.num_neg), device=x.device)
        for idx, data in enumerate(indices):
            rand_neg[idx] = x[data]

        del indices

        return rand_neg

    def paired_similarity(self, x, neg):
        x = x.unsqueeze(1)
        neg_similarity = torch.matmul(x, neg.transpose(-2, -1))  # (bhw, 1, d) * (bhw, d, k) -> (bhw, 1, k)
        neg_similarity = neg_similarity.squeeze(1)  # (bhw, k)
        neg_similarity = torch.exp(neg_similarity / self.temperature)

        return neg_similarity

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
        # flat_x1 = x1
        # flat_x2 = x2

        if self.cal_type == "random":
            neg = self.random(flat_x1)
        elif self.cal_type == "distance":
            neg = distance(flat_x1, b, self.num_neg)
        elif self.cal_type == "cosine":
            x1_norm = self._normalize(flat_x1)
            neg = self.cosine(x1_norm, self.num_neg)
        else:
            raise ValueError(f"No support {self.cal_type}")

        x1_norm = self._normalize(flat_x1)
        x2_norm = self._normalize(flat_x2)
        neg_norm = self._normalize(neg)

        pos_similarity = torch.multiply(x1_norm, x2_norm)
        pos_similarity = torch.exp(pos_similarity / self.temperature)
        neg_similarity = self.paired_similarity(x1_norm, neg_norm)

        positive = torch.sum(pos_similarity, dim=1)  # (bhw, )
        negative = torch.sum(neg_similarity, dim=1)  # (bhw, k) -> (bhw) #noqa

        loss = -(torch.log(positive) - torch.log(positive + negative))

        if self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "mean":
            loss = torch.mean(loss)

        del x1, flat_x1, x2, flat_x2, x1_norm, x2_norm, neg, neg_norm, negative, positive, pos_similarity, neg_similarity

        return loss


class ClusterLoss(nn.Module):
    def __init__(self,
                 temperature: float,
                 eps: float,
                 world_size: int = 4
                 ):
        super().__init__()
        self.temperature = temperature
        self.epsilon = eps
        self.world_size = world_size

    @torch.no_grad()
    def distributed_sinkhorn(self, out):  # TODO study about sinkhorn
        ''''
        out : (accum * 2bhw + 2bhw, num_prototypes) -> queue-weight scores + out_prototypes
        '''
        Q = torch.exp(out / self.epsilon).t()  # Q is K-by-B for consistency with notations from our paper
        # (accum * 2bhw + 2bhw, num_prototypes) -> (num_prototypes, accum * 2bhw + 2bhw)

        B = Q.shape[1] * self.world_size  # number of samples to assign
        # B = Q.shape[1] * args.world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        sum_Q = all_reduce_tensor(sum_Q)
        Q /= sum_Q

        for it in range(3):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            sum_of_rows = all_reduce_tensor(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment

        return Q.t()

    def forward(self, normalized_semantic_feat: torch.Tensor, out_prototypes: torch.Tensor, weight,
                queue: torch.Tensor):
        '''
        :param normalized_semantic_feat: (2bhw, hidden_dim)
        :param out_prototypes: (2bhw, num_prototypes)
        :param weight: Linear Classifier weight
        :param queue: (2bhw, num_prototypes) maybe same as out_prototypes?
        :return:
        '''
        with torch.no_grad():  # queue : stoarge of normalized semantic features (accum * 2 * bhw, hidden_dim)
            embedding = normalized_semantic_feat.detach()
            out = out_prototypes.detach()  # (2bhw, num_prototypes)

            bhw = out.shape[0]  # 2bhw
            if queue is not None:  # after queue_start_iter iterations
                out = torch.cat((torch.mm(queue, weight.t()),
                                 out))  # (accum * 2bhw, hidden_dim) * (hidden-dim, num_prototypes) -> (accum * 2bhw, num_prototypes)
                # (2bhw, num_prototypes) -> ( accum * 2bhw + 2bhw, num_prototypes)

                # fill the queue
                queue[bhw:] = queue[:-bhw].clone()
                queue[:bhw] = embedding

            q = self.distributed_sinkhorn(out)[-bhw:]  # (2bhw, num_prototypes)

        # cluster assignment prediction
        x = out_prototypes / self.temperature
        loss = -0.5 * torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))

        return loss, queue


class CLUBLoss(nn.Module):
    def __init__(self,
                 neg_sample: int = 100,
                 reduction: str = "mean"):
        super().__init__()
        self.num_neg = neg_sample

        self.reduction = reduction

    def forward(self, x: torch.Tensor, p_mu: torch.tensor, p_logvar: torch.tensor) -> torch.Tensor:
        '''

        :param x:  (b, d, h, w) -> (bhw, d)
        :param mu:
        :param logvar:
        :return:
        '''
        x = x.permute(0, 2, 3, 1).contiguous()
        b, h, w, d = x.shape
        flat_x = x.view(-1, d)  # (bhw, d)

        positive = -0.5 * torch.sum(
            torch.square(flat_x - p_mu) / torch.exp(p_logvar), dim=-1  # (bhw, d) -> (bhw)
        )
        # original code
        split_p_mu = torch.chunk(p_mu, chunks=h, dim=0)  # split tensor for code optimization
        split_p_logvar = torch.chunk(p_logvar, chunks=h, dim=0)  # split tensor for code optimization
        split_positive = torch.chunk(positive, chunks=h, dim=0)

        # neg = distance(flat_x, b, self.num_neg)  # (bhw, num_neg, d)

        # negative = -0.5 * torch.mean(
        #     torch.sum(
        #         torch.square(flat_x.unsqueeze(0) - p_mu.unsqueeze(1)) /
        #         torch.exp(p_logvar.unsqueeze(1)),
        #         dim=-1
        #     ),
        #     dim=-1
        # )
        #
        # loss = positive - negative
        # loss = torch.mean(loss)
        #
        # return loss
        loss = torch.tensor(0., device=x.device)

        for iter in range(h):
            p_mu_ = split_p_mu[iter]  # (bhw/h, d)
            p_logvar_ = split_p_logvar[iter]  # (bhw/h, d)

            negative = -0.5 * torch.mean(
                torch.sum(
                    torch.square(flat_x.unsqueeze(0) - p_mu_.unsqueeze(1)) /
                    torch.exp(p_logvar_.unsqueeze(1)),
                    dim=-1
                ),
                dim=-1
            )
            loss_ = torch.mean(split_positive[iter] - negative)  # bhw/28
            loss += loss_
            del loss_, negative, p_mu_, p_logvar_
        loss = loss / (h)

        del positive
        del split_positive, split_p_mu, split_p_logvar

        # if self.reduction == "sum":
        #     loss = torch.sum(loss)
        # elif self.reduction == "mean":
        #     loss = torch.mean(loss)

        return loss

        # positive = -0.5 * torch.sum(
        #     torch.square(flat_x - p_mu) / torch.exp(p_logvar), dim=-1  # (bhw, d) -> (bhw)
        # )
        # # p_mu = torch.chunk(p_mu, chunks=h*w, dim=0)  # split tensor for code optimization
        # # p_logvar = torch.chunk(p_logvar, chunks=h*w, dim=0)  # split tensor for code optimization
        # # positive = torch.chunk(positive, chunks=h*w, dim=0)
        # # negative = -0.5 * torch.mean(
        # #         torch.sum(
        # #             torch.square(flat_x.unsqueeze(0) - p_mu.unsqueeze(1)) /
        # #             torch.exp(p_logvar.unsqueeze(1)),
        # #             dim=-1
        # #         ),
        # #         dim=-1
        # #     )
        # loss = torch.tensor(0., device=x.device)
        # for iter in range(h * w):
        #     p_mu_ = p_mu[iter * b: (iter + 1) * b]
        #     p_logvar_ = p_logvar[iter * b: (iter + 1) * b]
        #     positive_ = positive[iter * b: (iter + 1) * b]
        #
        #     negative = -0.5 * torch.mean(
        #         torch.sum(
        #             torch.square(flat_x.unsqueeze(0) - p_mu_.unsqueeze(1)) /
        #             torch.exp(p_logvar_.unsqueeze(1)),
        #             dim=-1
        #         ),
        #         dim=-1
        #     )
        #     loss_ = positive_ - negative
        #     loss += loss_
        # loss = loss / (h * w)
        #
        # del positive, negative
        #
        # if self.reduction == "sum":
        #     loss = torch.sum(loss)
        # elif self.reduction == "mean":
        #     loss = torch.mean(loss)
        #
        # return loss


class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(MINE, self).__init__()
        self.T_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))

    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]

        T0 = self.T_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.T_func(torch.cat([x_samples, y_shuffle], dim=-1))

        lower_bound = T0.mean() - torch.log(T1.exp().mean())

        # compute the negative loss (maximise loss == minimise -loss)
        return - lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)


class NWJ(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(NWJ, self).__init__()
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))

    def forward(self, x_samples, y_samples):
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim=-1)) - 1.  # shape [sample_size, sample_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim=1) - np.log(sample_size)).exp().mean()
        return lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)


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
