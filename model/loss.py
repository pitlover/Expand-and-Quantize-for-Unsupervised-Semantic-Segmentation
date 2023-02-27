
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import pairwise_cosine_similarity
from utils.dist_utils import all_reduce_tensor
import numpy as np


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


class MarginRankingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.marginloss = nn.MarginRankingLoss(margin=0.0)

    def corr_matrix(self, x: torch.Tensor):
        x_flat = x.permute(0, 2, 3, 1).contiguous()
        # x_flat = x.permute(1, 2, 0).contiguous()
        x_flat = x_flat.reshape(-1, self.d)  # (hw, d)
        norm_x = F.normalize(x_flat, dim=1)  # (hw, d)

        return torch.matmul(norm_x, norm_x.T)

    def forward(self, ori, aug):
        '''

        :param ori: (b, hidden_dim, h, w)
        :param aug: (b, hidden_dim, h, w)
        :return:
        '''
        b, d, h, w = ori.shape
        self.d = d
        self.device = ori.device

        ori_corr = self.corr_matrix(ori)  # (hw, hw)
        aug_corr = self.corr_matrix(aug)  # (hw, hw)
        #
        # loss = ori_corr - aug_corr
        # loss = torch.mean(loss)

        # mask = (loss > 0.1).float()
        # loss *= mask
        # loss = torch.sum(loss) / max(1, torch.sum(mask))

        rank_input1 = ori_corr  # (hw, )
        rank_input2 = torch.roll(rank_input1, 1, 1)  # (hw, )
        rank_target, rank_margin = self.get_target_margin(aug_corr)
        rank_target_nonzero = rank_target.clone()
        rank_target_nonzero[rank_target_nonzero == 0] = 1
        rank_input2 = rank_input2 + rank_margin / rank_target_nonzero
        loss = self.marginloss(rank_input1, rank_input2, rank_target)

        loss = torch.mean(loss)

        return loss

    def get_target_margin(self, aug_corr):
        target1 = aug_corr.detach().cpu().numpy()
        target2 = torch.roll(aug_corr, 1, 1).detach().cpu().numpy()
        # target2 = torch.roll(aug_corr, 1, 0).detach().cpu().numpy()
        # target2 = torch.roll(aug_corr, -1).detach().cpu().numpy()

        greater = np.array(target1 > target2, dtype="float")
        less = np.array(target1 < target2, dtype="float") * (-1)

        target = greater + less
        target = torch.from_numpy(target).float().to(self.device)

        margin = abs(target1 - target2)
        margin = torch.from_numpy(margin).float().to(self.device)

        return target, margin


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


class ProxyLoss(nn.Module):
    def __init__(self,
                 normalize: str = "l2",
                 temperature: float = 1.0,
                 reduction: str = "mean",
                 num_queries: int = 50,
                 num_neg: int = 256,
                 cal_type: str = "random"):
        super().__init__()

        self.normalize = normalize
        self.temperature = temperature
        self.reduction = reduction
        self.cal_type = cal_type
        self.num_queries = num_queries
        self.num_neg = num_neg

    def forward(self, queue: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        '''

        :param queue: (n_classes, queue_size, hidden_dim)
        :param centroids: (n_classes, hidden_dim)
        :return:
        '''
        n_cluster = len(queue)
        loss = torch.tensor(0, device=centroids.device)

        for i in range(n_cluster):
            # select anchor pixel
            rand_idx = torch.randint(queue[i].shape[0], size=(self.num_queries,))
            query_idx = (queue[i][rand_idx].clone().cuda())

            # with torch.no_grad():
            centroid = (centroids[i].unsqueeze(0).unsqueeze(0).repeat(self.num_queries, 1,
                                                                      1))  # (num_queries, 1, hidden_dim)

            _i, i_ = queue[:i], queue[i + 1:]
            if len(_i) > 0:
                cat_i = torch.cat([_ for _ in _i], dim=0)
            else:
                cat_i = torch.tensor([], device=centroid.device)

            if len(i_) > 0:
                cat_i_ = torch.cat([_ for _ in i_], dim=0)
            else:
                cat_i_ = torch.tensor([], device=centroid.device)

            # TODO randomly choose neg vs uniformly choose neg per class
            neg_feat = torch.concat([cat_i, cat_i_], dim=0) # (all_negatives, hidden_dim)
            # randint
            neg_idx = torch.randint(neg_feat.shape[0], size=(self.num_queries * self.num_neg,),
                                    device=centroid.device)
            neg_feat = neg_feat[neg_idx].reshape(self.num_queries, self.num_neg, -1)
            all_feat = torch.cat((centroid, neg_feat), dim=1)  # (num_queries, 1+ neg, hidden_dim)

            logits = torch.cosine_similarity(query_idx.unsqueeze(1), all_feat, dim=2)
            loss = loss + F.cross_entropy(logits / self.temperature,
                                          torch.zeros(self.num_queries, device=centroids.device).long())

        return loss / n_cluster

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
        Q = Q / sum_Q

        for it in range(3):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            sum_of_rows = all_reduce_tensor(sum_of_rows)
            Q = Q / sum_of_rows
            Q = Q / K

            # normalize each column: total weight per sample must be 1/B
            Q = Q / torch.sum(Q, dim=0, keepdim=True)
            Q = Q / B

        Q = Q * B  # the colomns must sum to 1 so that Q is an assignment

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
        del q, x

        return loss, queue

    # def forward(self, normalized_semantic_feat: torch.Tensor, out_prototypes: torch.Tensor, weight,
    #             queue: torch.Tensor, neg_index):
    #     '''
    #     :param normalized_semantic_feat: (2bhw, hidden_dim)
    #     :param out_prototypes: (2bhw, num_prototypes)
    #     :param weight: Linear Classifier weight
    #     :param queue: (2bhw, num_prototypes) maybe same as out_prototypes?
    #     :return:
    #     '''
    #     # cluster assignment prediction
    #     x = out_prototypes / self.temperature
    #     scores = F.softmax(x, dim=1)  # (2bhw,)
    #     ori_scores, aug_scores = torch.chunk(scores, chunks=2, dim=0)  # (bhw, n_prototypes)
    #
    #     pos_similarity = torch.multiply(ori_scores, aug_scores)
    #     pos_similarity = torch.exp(pos_similarity / self.temperature)
    #     neg = F.embedding(neg_index, ori_scores)
    #     neg_similarity = self.paired_similiarty(ori_scores, neg)
    #
    #     positive = torch.sum(pos_similarity, dim=1)
    #     negative = torch.sum(neg_similarity, dim=1)
    #
    #     cluster_loss = -(torch.log(positive) - torch.log(positive + negative))
    #
    #     avg_p = aug_scores.mean(0)
    #     avg_entropy = -avg_p * torch.log(avg_p + 1e-8)
    #     avg_entropy = torch.sum(avg_entropy, dim=-1)  # (1,)
    #
    #     loss = torch.mean(cluster_loss) - self.epsilon * avg_entropy
    #
    #     return loss, queue


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


class EntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        '''

        :param p:
        :param q:
        :return:
        '''
        # TODO check only for ori?
        avg_p = p.mean(0)
        avg_entropy = -avg_p * torch.log(avg_p + 1e-8)
        avg_entropy = torch.sum(avg_entropy, dim=-1)
        return -avg_entropy


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


class JSDPosLoss(nn.Module):
    def __init__(self,
                 reduction: str = "batchmean",
                 num_query: int = 3,
                 num_pos: int = 10):
        super().__init__()
        self.kl = nn.KLDivLoss(reduction=reduction)
        self.num_query = num_query
        self.num_pos = num_pos

    def jsd(self, p: torch.Tensor, q: torch.Tensor):
        '''

        :param p: (b, selected, dim)
        :param q: (b, hw, dim)
        :return:
        '''

        m = torch.clamp((p + q) * 0.5, 1e-7, 1).log()  # (b, hw, selected, dim)
        loss = (self.kl(m, p) + self.kl(m, q)) * 0.5  # (b, hw, selected, dim)

        return loss

    def forward(self, z: torch.Tensor, z_pos: torch.Tensor, z_dis: torch.Tensor, z_pos_dis: torch.Tensor):
        '''

        :param z: (b, h, w, pq_dim)
        :param z_pos: (b, h, w, pq_dim)
        :param z_dis: (b, h, w, num_pq)
        :param z_pos_dis:  (b, h, w, num_pq)
        :return:
        '''
        # img <-> pos_img
        # coord_shape = [z.shape[0], 11, 11, 2]  # (b, 11, 11, 2)
        # coord = torch.rand(coord_shape, device=z.device) * 2 - 1
        # sample_z = sample(z, coord)  # (b, pq_dim, 11, 11) # TODO better way to select query points
        # sample_z_prob = sample(z_dis, coord) # (b, num_pq, 11, 11)
        #
        # attn = torch.einsum("nchw,ncij->nhwij", sample_z, z_pos)
        # attn -= attn.mean([3, 4], keepdim=True)
        # attn = attn.clamp(0)  # (b, 11, 11, h, w)
        # sample_z = torch.index_select
        # z_dis = z_dis.view()
        # b, d, h, w = z.shape
        # num_pq = z_dis.shape[1]
        #
        # z = z.permute(0, 2, 3, 1).reshape(b, -1, d) # (b, hw, d)
        # z_pos = z_pos.permute(0, 2, 3, 1).reshape(b, -1, d) # (b, hw, d)
        # z_dis = z_dis.permute(0, 2, 3, 1).reshape(b, -1, num_pq)  # (b, hw, d)
        # z_pos_dis = z_pos_dis.permute(0, 2, 3, 1).reshape(b, -1, num_pq)  # (b, hw, d)
        # loss = 0
        # count = 0
        # for i in range(b):
        #     rand_11 = torch.randint(0, h*w, (11,), device=z.device)
        #     sample_z = F.embedding(rand_11, z[i])
        #     sample_z_prob = F.embedding(rand_11, z_dis[i]) # (11, num_pq)
        #
        #     attn = torch.matmul(sample_z, z_pos[i].t()) ## (11, hw)
        #     attn -= attn.mean(dim=-1, keepdim=True)
        #     attn = attn.clamp(0)  # (11, hw)
        #
        #     for j in range(11):
        #         indices = torch.nonzero(attn[j])
        #         # sample_z_pos_prob = F.embedding(indices, z_pos_dis[i]) # (#nonzero, 1, num_pq)
        #         sample_z_pos_prob = F.embedding(indices, z_pos_dis[i]).squeeze(1) # (#nonzero, num_pq)
        #         for k in range(sample_z_pos_prob.shape[0]):
        #             loss += self.jsd(sample_z_prob, sample_z_pos_prob[k])
        #             count += 1
        # loss = loss / (count)
        #
        # return loss

        b, h, w, d = z.shape
        num_pq = z_dis.shape[-1]

        z = z.reshape(b, -1, d)  # (b, hw, d)
        z_pos = z_pos.reshape(b, -1, d)  # (b, hw, d)
        z_dis = z_dis.reshape(b, -1, num_pq)  # (b, hw, num_pq)
        z_pos_dis = z_pos_dis.reshape(b, -1, num_pq)  # (b, hw, num_pq)

        # select random 11 patches per image

        rand_11 = torch.randint(0, h * w, (b, self.num_query,), device=z.device)  # (b, num_query)
        batch = torch.arange(start=0, end=b * h * w, step=h * w, device=z.device).unsqueeze(-1)
        rand_11 = rand_11 + batch

        sample_z = F.embedding(rand_11, z.reshape(-1, d))  # (b, num_query, d)
        sample_z_dis = F.embedding(rand_11, z_dis.reshape(-1, num_pq))  # (b, num_query, num_pq)

        with torch.no_grad():
            attn = torch.einsum("bsc,bdc->bsd", sample_z, z_pos)  # (b, 11, hw)
            # attn -= attn.mean(dim=-1, keepdim=True)
            # attn = attn.clamp(0)
            attn = torch.topk(attn, k=self.num_pos, dim=-1).indices  # (b, 11, topk)
            # mask = (attn > 0)  # (b, num_query, hw)
            batch = torch.arange(start=0, end=b * h * w, step=h * w, device=z.device).unsqueeze(-1).unsqueeze(-1)
            batch_attn = attn + batch

            z_pos_dis_ = torch.index_select(z_pos_dis.reshape(-1, num_pq), 0, batch_attn.reshape(-1))
            sample_z_dis_ = sample_z_dis.unsqueeze(1).repeat(1, self.num_pos, 1, 1).reshape(-1, num_pq)

            #################
            # spatial_index = torch.arange(h * w).unsqueeze(0).unsqueeze(0).repeat(b, self.num_query, 1)[mask].tolist()
            # query_index = torch.arange(self.num_query).unsqueeze(0).unsqueeze(-1).repeat(b, 1, self.num_pos)[mask].tolist()
            # batch_index = torch.arange(b).reshape(b, 1, 1).repeat(1, self.num_query, self.num_pos)[mask].tolist()

        loss = self.jsd(sample_z_dis_, z_pos_dis_)
        # loss = self.jsd(z_pos_dis[batch_index, spatial_index, :], sample_z_dis[batch_index, query_index, :])
        #################

        # sample_z_dis = sample_z_dis.unsqueeze(-2).repeat(1, 1, h * w, 1)  # (b, num_query, hw, num_pq)
        # z_pos_dis = z_pos_dis.unsqueeze(1).repeat(1, self.num_query, 1, 1)  # (b, num_query, hw, num_pq)
        # loss = self.jsd(sample_z_dis[mask], z_pos_dis[mask])

        del sample_z, sample_z_dis, z_pos, z_pos_dis, attn, rand_11, z

        return loss


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


class STEGOLoss(nn.Module):

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
        coord_shape = [orig_feats.shape[0], self.cfg["feature_samples"], self.cfg["feature_samples"], 2]

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
                self.cfg["neg_inter_weight"] * neg_inter_loss.mean()
                )
