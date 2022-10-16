import random
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
import torchvision.transforms as transforms

from model.blocks.resnet import EncResBlock
from model.dino.dino_featurizer import DinoFeaturizer
from model.loss import InfoNCELoss, ClusterLoss

import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


class DINOCluster(nn.Module):
    def __init__(self, cfg: dict, cfg_loss: dict, world_size: int = 4):  # cfg["model"], cfg["loss"]
        super().__init__()
        self.cfg = cfg
        self.cfg_loss = cfg_loss

        self.extractor = DinoFeaturizer(cfg["pretrained"])
        self.feat_dim = self.extractor.n_feats  # 384
        self.semantic_dim = cfg.get("semantic_dim", self.feat_dim)
        self.local_dim = cfg.get("local_dim", self.feat_dim)
        self.hidden_dim = cfg.get("hidden_dim", self.feat_dim)

        # -------- semantic-encoder -------- #
        num_enc_blocks = cfg["enc_num_blocks"]
        semantic_enc_proj = []
        for i in range(num_enc_blocks):
            semantic_enc_proj.append(EncResBlock(self.feat_dim if (i == 0) else self.semantic_dim, self.semantic_dim))
        self.semantic_enc_proj = nn.Sequential(*semantic_enc_proj)

        # -------- prototype -------- #
        self.queue = None  # placeholder
        self.queue_first_time = True
        self.num_prototypes = cfg_loss["cluster"]["num_prototypes"]
        self.prototypes = nn.Linear(self.semantic_dim, self.num_prototypes, bias=False)
        # -------- loss -------- #
        self.infonce_loss = InfoNCELoss(normalize=self.cfg_loss["info_nce"].get("normalize", "l2"),
                                        neg_sample=self.cfg_loss["info_nce"].get("neg_sample", 0),
                                        temperature=self.cfg_loss["info_nce"].get("temperature", 1.0),
                                        cal_type=self.cfg_loss["info_nce"].get("cal_type", "random")
                                        )
        self.cluster_loss = ClusterLoss(temperature=self.cfg_loss["cluster"].get("temperature", 1.0),
                                        eps=self.cfg_loss["cluster"].get("eps", 0.001),
                                        world_size=world_size)
        # -------- t-sne -------- #
        self.iteration = 0
        self.tsne = cfg.get("tsne", False)

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

    def forward(self, img: torch.Tensor, queue: torch.Tensor = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        img_aug_1 = img  # (b, 3, h, w)
        img_aug_2 = self._photometric_aug(img)  # (b, 3, h, w)

        # TODO contrast geometric
        # seed = random.randint(0, 2147483647)
        # self._set_seed(seed)
        # img_aug_2 = self._geometric_aug(img_aug_2)  # (b, 3, h, w)

        img = torch.cat([img_aug_1, img_aug_2], dim=0)  # (2b, 3, h, w)
        dino_feat = self.extractor(img)  # (2b, 384, 28, 28) (2b, d, h/8, w/8)
        output = dict()

        semantic_feat = self.semantic_enc_proj(
            dino_feat)  # (2b, hidden_d, h, w)    # TODO maybe after noramlize for eval?

        semantic_feat_img1, semantic_feat_img2 = torch.chunk(semantic_feat, chunks=2, dim=0)  # (b, hidden_d, h, w)

        flat_semantic_feat = semantic_feat.permute(0, 2, 3, 1).contiguous()
        flat_semantic_feat = flat_semantic_feat.view(-1, semantic_feat.shape[1])

        normalized_semantic_feat = F.normalize(flat_semantic_feat, dim=1, p=2)  # TODO is it right ?
        out_prototypes = self.prototypes(normalized_semantic_feat)

        if self.training and queue is not None:
            if self.queue_first_time:
                self.queue = queue
                self.queue_first_time = False

        # normalize the prototypes
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.prototypes.weight.copy_(w)

        output["contra-loss-pos"] = self.infonce_loss(semantic_feat_img1, semantic_feat_img2)
        output["swav-loss"], queue = self.cluster_loss(normalized_semantic_feat,
                                                       out_prototypes,
                                                       self.prototypes.weight,
                                                       self.queue)
        if self.training:  # update queue only for training
            self.queue = queue
            '''
            T-SNE visualization
            '''
            if self.tsne and self.iteration % 1000 == 0:
                # ------------------------- #
                flat_ori_feat = semantic_feat_img1.permute(0, 2, 3, 1).contiguous().view(-1,
                                                                                         semantic_feat_img1.shape[1])
                cpu_flat_ori_feat = flat_ori_feat.clone().detach().cpu().numpy()  # (bhw, d)
                tsne_np1 = TSNE(n_components=2)
                semantic_np = tsne_np1.fit_transform(cpu_flat_ori_feat)

                plt.figure(figsize=(10, 10))
                plt.scatter(semantic_np[:, 0], semantic_np[:, 1])
                plt.savefig(f'./plot/swav_semantic/swav_semantic_{self.iteration}.png')

            self.iteration += 1

        return dino_feat, semantic_feat_img1, out_prototypes, output
