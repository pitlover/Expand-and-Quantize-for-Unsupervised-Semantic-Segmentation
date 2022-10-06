import random
from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
import torchvision.transforms as transforms
from model.dino.dino_featurizer import DinoFeaturizer
from model.blocks.resnet import EncResBlock, DecResBlock, LayerNorm2d
from model.loss import InfoNCELoss
from torch.nn.utils.clip_grad import clip_grad_norm_


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

    def forward(self, img: torch.Tensor, club_optimizer=None, scaler=None
                ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        img_aug_1 = img  # (b, 3, h, w)
        img_aug_2 = self._photometric_aug(img)  # (b, 3, h, w)

        # TODO contrast geometric
        # seed = random.randint(0, 2147483647)
        # self._set_seed(seed)
        # img_aug_2 = self._geometric_aug(img_aug_2)  # (b, 3, h, w)

        img = torch.cat([img_aug_1, img_aug_2], dim=0)  # (2b, 3, h, w)
        dino_feat = self.extractor(img)  # (2b, 384, 28, 28) (2b, d, h/8, w/8)
        output = dict()

        semantic_feat = self.semantic_enc_proj(dino_feat)  # (2b, hidden_d, h, w)
        semantic_feat_img1, semantic_feat_img2 = torch.chunk(semantic_feat, chunks=2, dim=0)  # (b, hidden_d, h, w)

        if self.training:
            output["contra-loss-pos"] = self.infonce_loss(semantic_feat_img1, semantic_feat_img2)

        return dino_feat, semantic_feat_img1, output
