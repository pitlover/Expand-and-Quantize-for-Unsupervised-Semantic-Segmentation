import random
from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
import torchvision.transforms as transforms
from model.dino.dino_featurizer import DinoFeaturizer
from model.blocks.resnet_linear import EncResBlock, DecResBlock, LayerNorm2d
from model.loss import InfoNCELoss
from torch.nn.utils.clip_grad import clip_grad_norm_
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


class DINOCluster(nn.Module):
    def __init__(self, cfg: dict, cfg_loss: dict):  # cfg["model"], cfg["loss"]
        super().__init__()
        self.cfg = cfg
        self.cfg_loss = cfg_loss

        self.extractor = DinoFeaturizer(cfg["pretrained"])
        self.feat_dim = self.extractor.n_feats  # 384
        self.semantic_dim = cfg.get("semantic_dim", self.feat_dim)
        self.local_dim = cfg.get("local_dim", self.feat_dim)
        self.hidden_dim = cfg.get("hidden_dim", self.feat_dim)

        self.kmeans_init = cfg["k_means"]["init"]
        self.kmeans_n_cluster = cfg["k_means"]["n_cluster"]
        self.kmeans_n_pos = cfg["k_means"]["n_pos"]

        # -------- semantic-encoder -------- #
        num_enc_blocks = cfg["enc_num_blocks"]
        semantic_enc_proj = []
        for i in range(num_enc_blocks):
            semantic_enc_proj.append(EncResBlock(self.feat_dim if (i == 0) else self.semantic_dim, self.semantic_dim))
        self.semantic_enc_proj = nn.Sequential(*semantic_enc_proj)

        # -------- loss -------- #
        self.infonce_loss = InfoNCELoss(normalize=self.cfg_loss["info_nce"].get("normalize", "l2"),
                                        neg_sample=self.cfg_loss["info_nce"].get("neg_sample", 0),
                                        temperature=self.cfg_loss["info_nce"].get("temperature", 1.0),
                                        cal_type=self.cfg_loss["info_nce"].get("cal_type", "random")
                                        )
        self.iteration = 0

    def _photometric_aug(self, x: torch.Tensor):
        # b, 3, h, w = x.shape
        batch_size = x.shape[0]
        device = x.device

        # TODO check the power of augmentation
        random_scale = torch.ones(batch_size, 3, 1, 1, dtype=torch.float32, device=device).uniform_(0.9,
                                                                                                    1.1)  # noqa # color
        random_offset = torch.ones(batch_size, 3, 1, 1, dtype=torch.float32, device=device).uniform_(-0.1, 0.1)  # noqa
        x_aug = x * random_scale + random_offset

        if random.randint(0, 3) == 0:  # 25%
            x_aug = transforms.GaussianBlur(kernel_size=3)(x_aug)  # texture

        '''
        x_aug = transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomGrayscale(0.2),
                transforms.RandomApply([T.GaussianBlur((5, 5))])
            ])
        '''
        return x_aug

    def _geometric_aug(self, x: torch.Tensor):
        x_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=224 if self.training else 320, scale=(0.8, 1.0))
        ])

        return x_aug(x)

    @staticmethod
    def _set_seed(seed):
        random.seed(seed)  # apply this seed to img transforms
        torch.manual_seed(seed)  # needed for torchvision 0.7

    def forward(self, img: torch.Tensor, label: torch.Tensor = None, stage: int = 0
                ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        output = dict()
        minimum, total = self.kmeans_n_pos, 0
        if stage == 1:
            # photometric aug
            img_aug_1 = img
            img_aug_2 = self._photometric_aug(img)
            img = torch.cat([img_aug_1, img_aug_2], dim=0)  # (2b, 3, h, w)

            before_dino_feat = self.extractor(img)  # (2b, d, h, w)
            b, d, h, w = before_dino_feat.shape
            before_dino_feat = before_dino_feat.permute(0, 2, 3, 1).contiguous()  # (b, h, w, d)
            before_dino_feat = before_dino_feat.view(-1, d)  # (bhw, d)

            ori_dino_feat, aug_dino_feat = torch.chunk(before_dino_feat, chunks=2, dim=0)
            ori_dino_feat = ori_dino_feat.cpu().numpy()
            aug_dino_feat = aug_dino_feat.cpu().numpy()

            clustering = KMeans(init=self.kmeans_init, n_clusters=self.kmeans_n_cluster, random_state=0)
            clustering.fit(ori_dino_feat)  # ()
            centroids = np.array(clustering.cluster_centers_)  # (kmeans_n_cluster, d)
            clusters_labels = clustering.labels_.tolist()

            for i in range(self.kmeans_n_cluster):
                center = centroids[i].reshape(1, -1)  # (1, d)
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

            dino_feat = torch.cat([pos_ori_feat, pos_aug_feat], dim=0)

            semantic_feat = self.semantic_enc_proj(dino_feat)  # (2b, hidden_d, h, w)
            semantic_feat_img1, semantic_feat_img2 = torch.chunk(semantic_feat, chunks=2, dim=0)  # (b, hidden_d, h, w)

            if self.training:
                output["contra-loss-pos"] = self.infonce_loss(semantic_feat_img1, semantic_feat_img2)
            semantic_feat = semantic_feat_img1
            print("MINIMUM :", minimum, "TOTAL :", total)

            '''
            T-SNE visualization
            '''
            if self.iteration % 100 == 0:
                resize_label = label.unsqueeze(1)
                resize_label = resize_label.to(torch.float32)
                max_pool = nn.MaxPool2d(8, stride=8)
                resize_label = max_pool(resize_label)
                resize_label = resize_label.permute(0, 2, 3, 1).contiguous()  # (b, h, w, d)
                resize_label = resize_label.view(-1, 1)  # (bhw, 1)
                plt.figure(figsize=(10, 10))
                colors = ['bisque', 'forestgreen', 'slategrey', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple',
                          'yellow', 'chocolate', 'coral', 'orchid', 'steelblue', 'lawngreen', 'lightsalmon', 'hotpink',
                          "#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525",
                          "#A83683", "#4E655E", "#853541", "#3A3120", "#535D8E"]

                tsne_np = TSNE(n_components=2)
                dino_np = tsne_np.fit_transform(ori_dino_feat)

                plt.figure(figsize=(10, 10))
                for i in range(len(dino_np)):
                    if int(resize_label[i][0].item()) == -1:
                        continue
                    plt.scatter(dino_np[i][0], dino_np[i][1], color=colors[int(resize_label[i][0].item())])

                plt.savefig(f'./plot/ori/ori_{self.iteration}.png')

                # ------------------------- #
                cpu_semantic_feat = semantic_feat.detach().cpu().numpy()
                tsne_np1 = TSNE(n_components=2)
                semantic_np = tsne_np1.fit_transform(cpu_semantic_feat)

                plt.figure(figsize=(10, 10))
                plt.scatter(semantic_np[:, 0], semantic_np[:, 1])
                plt.savefig(f'./plot/semantic/semantic_{self.iteration}.png')

                # ------------------------- #
                tsne_np2 = TSNE(n_components=2)
                center_np = tsne_np2.fit_transform(centroids)
                plt.figure(figsize=(10, 10))
                plt.scatter(center_np[:, 0], center_np[:, 1])
                plt.savefig(f'./plot/center/center_{self.iteration}.png')

            self.iteration += 1
            del clustering, clusters_labels, data_within_cluster, data_idx, centroids, center
        else:
            dino_feat = self.extractor(img)  # (2b, d, h, w)
            b, d, h, w = dino_feat.shape
            dino_feat = dino_feat.permute(0, 2, 3, 1).contiguous()  # (b, h, w, d)
            dino_feat = dino_feat.view(-1, d)  # (bhw, d)

            semantic_feat = self.semantic_enc_proj(dino_feat)  # (bhw, d)
            semantic_feat = semantic_feat.reshape(b, h, w, -1)
            semantic_feat = semantic_feat.permute(0, 3, 1, 2).contiguous()
        return dino_feat, semantic_feat, output
