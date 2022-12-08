from typing import Dict, Tuple, List

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from model.blocks.module import SegmentationHead
from model.dino import DinoFeaturizer
from model.loss import STEGOLoss, ProxyLoss
from utils.dist_utils import all_reduce_tensor


class DIONEMA(nn.Module):
    def __init__(self, cfg: dict, cfg_loss: dict):
        # cfg : cfg["model"]
        super().__init__()
        self.cfg = cfg
        self.cfg_loss = cfg_loss

        self.extractor = DinoFeaturizer(cfg["pretrained"])
        self.feat_dim = self.extractor.n_feats  # 384
        self.hidden_dim = cfg["hidden_dim"]  # 70
        self.is_dropout = cfg["pretrained"]["dropout"]
        self.dropout = nn.Dropout2d(p=cfg["pretrained"]["drop_prob"])
        self.m = cfg["encoder"]["momentum"]  # TODO cosine annealing scheduling
        self.ts = cfg["encoder"]["temperature"]
        self.n_cluster = cfg["memory_bank"]["n_cluster"]
        self.num_support = cfg["memory_bank"]["num_support"]

        #  -------- student head -------- #
        self.b, self.d, self.h, self.w = None, None, None, None  # placeholder
        self.trainable_head = SegmentationHead(self.feat_dim, self.hidden_dim)

        #  -------- ema head -------- #
        self.ema_head = SegmentationHead(self.feat_dim, self.hidden_dim)

        for param_top, param_bottom in zip(self.trainable_head.parameters(), self.ema_head.parameters()):
            param_bottom.data.copy_(param_top.data)  # initialize
            param_bottom.requires_grad = False  # not update by gradient

        #  -------- memory bank -------- #
        self.margin = cfg["memory_bank"]["margin"]
        self.queue_size = cfg["memory_bank"]["queue_size"]
        self.queue_ptr = torch.zeros(self.n_cluster, 1, dtype=torch.long)
        self.need_initialize = True
        self.centroid = nn.Embedding(self.n_cluster, self.hidden_dim)
        # self.register_buffer("queue", torch.randn(self.n_cluster, self.queue_size, self.hidden_dim))  # (27, 256, 70)
        self.queue = []

        for i in range(self.n_cluster):
            self.queue.append([torch.zeros(0, self.hidden_dim)])

        # -------- loss -------- #
        # self.stego_loss = STEGOLoss(cfg=self.cfg_loss["stego"])
        self.info_nce = ProxyLoss(temperature=cfg_loss["info_nce"]["temperature"],
                                  num_queries=cfg_loss["info_nce"]["num_queries"],
                                  num_neg=cfg_loss["info_nce"]["num_neg"])

    def _flatten(self, x):
        '''

        :param x: (b, d, h, w)
        :return:  (bhw, d)
        '''
        # b, d, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # (b, h, w, d)
        flat_x = x.view(-1, self.d)  # (bhw, d)

        return flat_x

    def _reshape(self, x):
        '''

        :param x:  (bhw, d)
        :return:  (b, d, h, w)
        '''
        x = x.view(-1, self.h, self.w, self.d)
        reshape_x = x.permute(0, 3, 1, 2).contiguous()

        return reshape_x

    def _init_memory_bank(self, x):
        '''
        initialization of memory bank
        kmeans into n-classes and bring the n-support the closest sample
        :return:
        '''
        x = self._flatten(x)
        kmeans = faiss.Kmeans(d=self.hidden_dim, k=self.n_cluster, verbose=True, gpu=True)
        kmeans.min_points_per_centorids = self.num_support
        cpu_x = x.detach().cpu().numpy().astype(np.float32)
        kmeans.train(cpu_x)
        query_vector = kmeans.centroids  # (n_cluster, hidden_dim)

        index = faiss.IndexFlatL2(cpu_x.shape[-1])
        index.add(cpu_x)
        _, idx = index.search(query_vector, self.num_support)  # (n_cluster, n_kmeans_pos)

        idx = torch.from_numpy(idx)  # (n_cluster, num_support)
        idx = idx.reshape(-1)
        selected_support_ = x[idx]  # (n_cluster * num_support, hidden_dim)
        selected_support = selected_support_.view(-1, self.num_support,
                                                  self.hidden_dim)  # (n_cluster, num_support, hidden_dim)
        # centroids = torch.from_numpy(query_vector).to(x.device)
        centroids = torch.mean(selected_support, dim=1)  # (n_cluster, 1, hidden_dim) # TODO check centroids
        self.centroid.weight.data.copy_(centroids)

        for i in range(self.n_cluster):
            self.queue[i] = selected_support[i].detach().clone().cpu()

        '''
        T-SNE visualization
        '''

        # from sklearn.manifold import TSNE
        # from matplotlib import pyplot as plt
        #
        # tsne_np1 = TSNE(n_components=2)
        # total = torch.cat([centroids, selected_support_], dim=0)
        # fit_total = tsne_np1.fit_transform(total.detach().cpu().numpy())
        # center = fit_total[:27]
        # support = fit_total[27:]
        # # support = tsne_np1.fit_transform(selected_support_.detach().cpu().numpy())
        #
        # index = range(27)
        # index_ = [[a] * self.num_support for a in range(27)]
        # index_ = [j for sub in index_ for j in sub]
        # plt.figure(figsize=(10, 10))
        # plt.scatter(support[:, 0], support[:, 1], c=index_, cmap=plt.cm.hsv)
        # plt.scatter(center[:, 0], center[:, 1], s=100, c=index, marker="+", cmap=plt.cm.hsv)
        # for i, txt in enumerate(index):
        #     plt.annotate(f"{txt}", (center[i, 0], center[i, 1]))
        # for i, txt in enumerate(index_):
        #     plt.annotate(f"{txt}", (support[i, 0], support[i, 1]))
        #
        # plt.savefig(f'./plot/proxy/total_{self.num_support}.png')
        # exit()

    @torch.no_grad()
    def _momentum_update_ema_head(self):
        """
        Momentum update of the ema encoder
        """
        for param_q, param_k in zip(self.trainable_head.parameters(), self.ema_head.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def _normalize(self, x: torch.Tensor):
        '''

        :param x: (b, d, h, w)
        :return: (bhw, d)
        '''

        x = self._flatten(x)
        norm_x = F.normalize(x, dim=-1)

        return norm_x

    def _update_queue(self, x, norm_x):
        '''
        Dequeue and Enqueue
        : x : logits of teacher network :(b, d, h, w) -> (bhw, d)
        :return:
        '''

        centroid_norm = F.normalize(self.centroid.weight, dim=-1)

        distance = torch.sum(norm_x ** 2, dim=1, keepdim=True) + \
                   torch.sum(centroid_norm ** 2, dim=1) - \
                   2 * (torch.matmul(norm_x, centroid_norm.t()))  # (bhw, n_prototypes)
        idx = torch.argmin(distance, dim=-1)  # (bhw)
        idx_top2 = torch.topk(-distance, k=2, dim=-1)  # (bhw, 2)
        flat_raw = self._flatten(x)

        for i in range(self.n_cluster):
            mask = torch.eq(idx, i)  # (bhw, )
            above_mrg = (idx_top2.values[mask][:, 0] - idx_top2.values[mask][:, 1]) > self.margin  # TODO check margin
            selected_support = flat_raw[mask][above_mrg]
            # dequeue & enqueue
            self._dequeue_and_enqueue(keys=selected_support, idx=i)

    @torch.no_grad()
    def gather_together(self, data):
        dist.barrier()

        world_size = dist.get_world_size()
        gather_data = [None for _ in range(world_size)]
        dist.all_gather_object(gather_data, data)

        return gather_data

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, idx):
        '''

        :param keys: (selected_num, hidden_dim)
        :param queue: (num_support, hidden_dim)
        :param queue_ptr: int
        :param queue_size: queue_size
        :return:
        '''
        # gather keys before updating queue
        keys = keys.detach().clone().cpu()
        gathered_list = self.gather_together(keys)
        keys = torch.cat(gathered_list, dim=0).cuda()
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr[idx])

        queue = torch.cat((self.queue[idx], keys.cpu()), dim=0)
        if queue.shape[0] >= self.queue_size:
            self.queue[idx] = queue[-self.queue_size:, :]
            ptr = self.queue_size
        else:
            self.queue[idx] = queue
            ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[idx] = ptr

    def forward(self, img: torch.Tensor,
                aug_img: torch.Tensor = None,
                img_pos: torch.Tensor = None,
                label: torch.Tensor = None,
                it: int = 0
                ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        '''

        :param img: (b, d, h, w)
        :param aug_img: (b, d, h, w)
        :param img_pos: (b, d, h, w)
        :param label: (b, 1, h, w)
        :param it: iteration
        :return:
        '''
        outputs = {}

        # TODO geometric transform
        dino_feat_ori = self.extractor(img)  # (b, 384, 28, 28) (b, d, h, w)
        dino_feat_ori = self.dropout(dino_feat_ori)

        dino_feat_aug = self.extractor(aug_img)  # (b, 384, 28, 28) (b, d, h, w)
        dino_feat_aug = self.dropout(dino_feat_aug)

        # ---- trainable(ori) vs ema(aug) ---- #
        z1_1 = self.trainable_head(dino_feat_ori)
        self.b, self.d, self.h, self.w = z1_1.shape
        norm_z1_1 = self._normalize(z1_1)

        with torch.no_grad():
            self._momentum_update_ema_head()
            z1_2 = self.ema_head(dino_feat_aug)
            norm_z1_2 = self._normalize(z1_2).clone().detach()  # detach the gradient z2

        loss1 = F.mse_loss(norm_z1_1, norm_z1_2)
        outputs["mse-loss"] = loss1

        # initialize if needed
        if self.need_initialize:
            self.need_initialize = False
            self._init_memory_bank(z1_1)

        # update queue
        self._update_queue(z1_1, norm_z1_1)
        # TODO check if it normalize,
        # TODO cur_status: compare with normalize but update with nonnormalized

        # for i in range(self.n_cluster):
        #     print(len(self.queue[i]), f"-> {len(self.queue[i]) - self.num_support}")

        # update centroid
        outputs["info_nce"] = self.info_nce(self.queue, self.centroid)
        # cross loss

        # ---- stego-loss ---- #
        # if self.training:
        #     dino_feat_pos = self.extractor(img_pos)  # (b, 384, 28, 28) (b, d, h, w)
        #     dino_feat_pos = self.dropout(dino_feat_pos)
        #     z_pos = self.trainable_head(dino_feat_pos)
        #     outputs["stego-loss"] = self.stego_loss(dino_feat_ori, dino_feat_pos, z1_1, z_pos)

        # # TODO symmetric loss
        # # ---- trainable(aug) vs ema(ori) ---- #
        # z2_1 = self.trainable_head(dino_feat_aug)
        # norm_z2_1 = self._normalize(z2_1)
        #
        # with torch.no_grad():
        #     z2_2 = self.ema_head(dino_feat_ori)
        #     norm_z2_2 = self._normalize(z2_2).clone().detach()  # detach the gradient z2
        #
        # loss2 = F.mse_loss(norm_z2_1, norm_z2_2)
        #
        # outputs["mse1"] = loss1.mean()
        # outputs["mse2"] = loss2.mean()
        # outputs["mse-loss"] = (loss1 + loss2).mean()

        # ---- reshape ---- #
        out = self._reshape(norm_z1_1)

        return out, [z1_1, z1_2], outputs
