import random
from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from model.dino.dino_featurizer import DinoFeaturizer
from model.blocks.resnet_linear import EncResBlock, DecResBlock, LayerNorm2d
from model.loss import JSDLoss
from model.quantizer import VectorQuantizer, EMAVectorQuantizer, ProductQuantizerWrapper

import torchvision.transforms as transforms
from sklearn.cluster import KMeans
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


class DINOContra(nn.Module):
    def __init__(self, cfg: dict):  # cfg["model"]
        super().__init__()
        self.cfg = cfg

        self.extractor = DinoFeaturizer(cfg["pretrained"])
        self.feat_dim = self.extractor.n_feats  # 384
        self.hidden_dim = cfg.get("hidden_dim", self.feat_dim)

        self.kmeans_init = cfg["k_means"]["init"]
        self.kmeans_n_cluster = cfg["k_means"]["n_cluster"]
        self.kmeans_n_pos = cfg["k_means"]["n_pos"]

        self.iteration = 0
        # -------- encoder -------- #
        num_enc_blocks = cfg["enc_num_blocks"]
        enc_proj = []
        for i in range(num_enc_blocks):
            enc_proj.append(EncResBlock(self.feat_dim if (i == 0) else self.hidden_dim, self.hidden_dim))
        self.enc_proj = nn.Sequential(*enc_proj)

        # -------- vq -------- #
        vq_num_codebooks = cfg["vq"]["num_codebooks"]
        vq_embed_dims = cfg["vq"]["embed_dims"]
        assert len(vq_num_codebooks) == len(vq_embed_dims)
        self.num_vq = len(vq_num_codebooks)
        self.beta = cfg["vq"]["beta"]
        self.normalize = cfg["vq"]["normalize"]
        self.vq_type = cfg["vq"]["vq_type"]
        self.use_restart = cfg["vq"].get("use_restart", False)
        self.use_split = cfg["vq"].get("use_split", False)
        self.use_gumbel = cfg["vq"].get("use_gumbel", False)
        self.use_weighted_sum = cfg["vq"].get("use_weighted_sum", False)
        self.need_initialized = cfg["vq"].get("need_initialized", False)
        self.jsd = JSDLoss()

        self.num_pq = cfg["vq"].get("num_pq", 1)
        if isinstance(self.num_pq, int):
            self.num_pq = [self.num_pq] * self.num_vq

        vq_kwargs = dict(beta=self.beta, normalize=self.normalize,
                         use_restart=self.use_restart, use_gumbel=self.use_gumbel, use_split=self.use_split,
                         use_weighted_sum=self.use_weighted_sum, need_initialized=self.need_initialized)

        if self.vq_type == "ema":
            vq_kwargs["decay"] = cfg["vq"]["decay"]
            vq_kwargs["eps"] = cfg["vq"]["eps"]
            vq_blocks = [
                EMAVectorQuantizer(vq_num_codebooks[i], vq_embed_dims[i], **vq_kwargs) if (self.num_pq == 1) else
                ProductQuantizerWrapper(self.num_pq[i], vq_num_codebooks[i], vq_embed_dims[i], **vq_kwargs,
                                        quantizer_cls=EMAVectorQuantizer)
                for i in range(self.num_vq)
            ]
        elif self.vq_type == "param":
            vq_blocks = [
                VectorQuantizer(vq_num_codebooks[i], vq_embed_dims[i], **vq_kwargs) if (self.num_pq == 1) else
                ProductQuantizerWrapper(self.num_pq[i], vq_num_codebooks[i], vq_embed_dims[i], **vq_kwargs,
                                        quantizer_cls=VectorQuantizer)
                for i in range(self.num_vq)
            ]
        else:
            raise ValueError(f"Unsupported vq type {self.vq_type}.")
        self.vq_blocks = nn.ModuleList(vq_blocks)

        # -------- vq connections -------- #
        vq_input_proj = []
        for i in range(self.num_vq):
            vq_input_proj.append(nn.Sequential(
                nn.LeakyReLU(0.1, inplace=False),  # MOVED TO HERE
                # nn.Conv2d(self.hidden_dim, vq_embed_dims[i], 1, 1, 0, bias=False),
                nn.Linear(self.hidden_dim, vq_embed_dims[i]),
            ))
        self.vq_input_proj = nn.ModuleList(vq_input_proj)

        vq_output_proj = []
        for i in range(self.num_vq - 1):
            vq_output_proj.append(nn.Sequential(
                # nn.Conv2d(self.hidden_dim + vq_embed_dims[i], self.hidden_dim, 1, 1, 0),
                nn.Linear(self.hidden_dim + vq_embed_dims[i], self.hidden_dim),
                # nn.ReLU(inplace=True)  # ORIGINALLY HERE
                # nn.LeakyReLU(0.1, inplace=True)
            ))
        self.vq_output_proj = nn.ModuleList(vq_output_proj)

        self.agg_type = cfg["vq"].get("agg_type", "concat")
        if (self.agg_type == "cat") or (self.agg_type == "concat"):
            self.agg_type = "concat"
            # self.vq_aggregate_proj = nn.Conv2d(sum(vq_embed_dims), self.hidden_dim, 1, 1, 0)
            self.vq_aggregate_proj = nn.Linear(sum(vq_embed_dims), self.hidden_dim)
        elif (self.agg_type == "add") or (self.agg_type == "sum"):
            self.agg_type = "add"
            self.vq_aggregate_proj = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1, 1, 0)
        else:
            raise ValueError(f"Unsupported aggregate type {self.agg_type}.")

        # -------- decoder -------- #
        num_dec_blocks = cfg["dec_num_blocks"]
        dec_proj = []
        for i in range(num_dec_blocks):
            dec_proj.append(
                DecResBlock(self.hidden_dim, self.feat_dim if (i == num_dec_blocks - 1) else self.hidden_dim))
        self.dec_proj = nn.Sequential(*dec_proj)

        last_norm = cfg.get("last_norm", False)
        self.dec_norm = LayerNorm2d(self.feat_dim) if last_norm else None

    def _photo_aug(self, x: torch.Tensor):
        # b, 3, h, w = x.shape
        batch_size = x.shape[0]
        device = x.device
        random_scale = torch.ones(batch_size, 3, 1, 1, dtype=torch.float32, device=device).uniform_(0.9,
                                                                                                    1.1)  # noqa # color
        random_offset = torch.ones(batch_size, 3, 1, 1, dtype=torch.float32, device=device).uniform_(-0.1, 0.1)  # noqa
        x_aug = x * random_scale + random_offset

        if random.randint(0, 3) == 0:  # 25%
            x_aug = transforms.GaussianBlur(kernel_size=3)(x_aug)  # texture
        return x_aug

    def forward(self, img: torch.Tensor, stage: int = 0
                ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:

        minimum, total = self.kmeans_n_pos, 0
        if stage == 1:
            # photometric aug
            img_aug_1 = img
            img_aug_2 = self._photo_aug(img)
            img = torch.cat([img_aug_1, img_aug_2], dim=0)  # (2b, 3, h, w)
            before_dino_feat = self.extractor(img)  # (2b, d, h, w)

            b, d, h, w = before_dino_feat.shape
            before_dino_feat = before_dino_feat.permute(0, 2, 3, 1).contiguous()  # (2b, h, w, d)
            before_dino_feat = before_dino_feat.view(-1, d)  # (bhw, d)

            ori_dino_feat, aug_dino_feat = torch.chunk(before_dino_feat, chunks=2, dim=0)

            ori_dino_feat = ori_dino_feat.cpu().numpy()
            aug_dino_feat = aug_dino_feat.cpu().numpy()

            clustering = KMeans(init=self.kmeans_init, n_clusters=self.kmeans_n_cluster, random_state=0).fit(
                ori_dino_feat)  # ()
            centroids = np.array(clustering.cluster_centers_)  # (kmeans_n_cluster, d)
            clusters_labels = clustering.labels_.tolist()

            for i in range(self.kmeans_n_cluster):
                center = centroids[i].reshape(1, -1)  # (d,)
                data_within_cluster = [idx for idx, clu_num in enumerate(clusters_labels) if clu_num == i]
                if len(data_within_cluster) < self.kmeans_n_pos:
                    current_n_pos = len(data_within_cluster)
                    if current_n_pos < minimum:
                        minimum = current_n_pos
                else:
                    current_n_pos = self.kmeans_n_pos
                total += (current_n_pos)

                ori_matrix = np.zeros((len(data_within_cluster), centroids.shape[1]))
                aug_matrix = np.zeros((len(data_within_cluster), centroids.shape[1]))
                # one_cluster_tf_matrix = np.zeros((len(data_within_cluster), centroids.shape[1]))

                for row_num, data_idx in enumerate(data_within_cluster):
                    ori_matrix[row_num] = ori_dino_feat[data_idx]
                    aug_matrix[row_num] = aug_dino_feat[data_idx]

                center = torch.from_numpy(center).float().to(img.device)
                ori_matrix = torch.from_numpy(ori_matrix).float().to(img.device)
                aug_matrix = torch.from_numpy(aug_matrix).float().to(img.device)

                distance_ = torch.cdist(center, ori_matrix)
                distance_index = torch.topk(distance_, current_n_pos).indices  # (1, n_pos)
                pos_ori_dino_feat_ = F.embedding(distance_index, ori_matrix).squeeze(0)  # (1, n_pos, d) -> (n_pos, d)
                pos_aug_dino_feat_ = F.embedding(distance_index, aug_matrix).squeeze(0)

                if i == 0:
                    pos_ori_feat = pos_ori_dino_feat_
                    pos_aug_feat = pos_aug_dino_feat_
                else:
                    pos_ori_feat = torch.cat([pos_ori_feat, pos_ori_dino_feat_], dim=0)
                    pos_aug_feat = torch.cat([pos_aug_feat, pos_aug_dino_feat_], dim=0)

            dino_feat = torch.cat([pos_ori_feat, pos_aug_feat], dim=0)  # (2 * k_cluster * n_pos , hidden_dim)
            print(f"[{self.iteration}] MINIMUM : {minimum}, TOTAL : {total}")

            self.iteration += 1
            del clustering, clusters_labels, data_within_cluster, data_idx, centroids, center

        else:
            img_aug_1 = img
            img_aug_2 = self._photo_aug(img)
            img = torch.cat([img_aug_1, img_aug_2], dim=0)  # (2b, 3, h, w)
            dino_feat = self.extractor(img)  # (2b, d, h, w)

            b, d, h, w = dino_feat.shape
            dino_feat = dino_feat.permute(0, 2, 3, 1).contiguous()  # (b, h, w, d)
            dino_feat = dino_feat.view(-1, d)  # (bhw, d)

        feat = self.enc_proj(dino_feat)
        output = dict()
        feat_vqs = []

        vq_top_dis_prob = None  # placeholder
        vq_bottom_dis_prob = None  # placeholder

        for i in range(self.num_vq):
            feat_i = self.vq_input_proj[i](feat)
            feat_vq_i, vq_i_output, dis_prob = self.vq_blocks[i](feat_i)

            if i == 0:
                vq_top_dis_prob = dis_prob
            elif i == self.num_vq - 1:
                vq_bottom_dis_prob = dis_prob

            feat_vqs.append(feat_vq_i)

            for k, v in vq_i_output.items():
                output[f"vq{i}-{k}"] = v

            if i < self.num_vq - 1:
                feat_i = torch.cat([feat, feat_vq_i], dim=1)
                feat = self.vq_output_proj[i](feat_i)

        if self.agg_type == "concat":
            feat = torch.cat(feat_vqs, dim=1)  # (2 * k_cluster * n_pos, hidden + hidden)
        elif self.agg_type == "add":
            feat = sum(feat_vqs)
        else:
            raise ValueError
        feat = self.vq_aggregate_proj(feat)  # (2 * k_cluster * n_pos, hidden)
        recon = self.dec_proj(feat)  # (2 * k_cluster * n_pos, d)

        recon_loss = F.mse_loss(recon, dino_feat)

        output["recon-loss"] = recon_loss

        # contrastive loss
        top_dis_prob_1, top_dis_prob_2 = torch.chunk(vq_top_dis_prob, chunks=2, dim=0)  # (2bhw, K) -> (2, bhw, K)
        output["contra-loss-pos"] = self.jsd(top_dis_prob_1, top_dis_prob_2)

        bottom_dis_prob_1, bottom_dis_prob_2 = torch.chunk(vq_bottom_dis_prob, chunks=2, dim=0)
        output["contra-loss-neg"] = self.jsd(bottom_dis_prob_1, bottom_dis_prob_2)

        # split half
        feat = torch.chunk(feat, chunks=2, dim=0)[0]
        feat_vqs = [torch.chunk(vq_i, chunks=2, dim=0)[0] for vq_i in feat_vqs]

        '''
        T-SNE visualization
        '''
        if self.training and self.iteration % 100 == 0:
            # ------------------------- #
            cpu_ori_quantized = feat_vqs[0].detach().cpu().numpy()
            tsne_np1 = TSNE(n_components=2)
            semantic_np = tsne_np1.fit_transform(cpu_ori_quantized)

            plt.figure(figsize=(10, 10))
            plt.scatter(semantic_np[:, 0], semantic_np[:, 1])
            plt.savefig(f'./plot/vq_semantic/vq_semantic_{self.iteration}.png')

        return feat, feat_vqs, output
