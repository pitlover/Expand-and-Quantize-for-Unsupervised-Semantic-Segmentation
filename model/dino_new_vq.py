from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.dino import DinoFeaturizer
from model.blocks.resnet import EncResBlock, DecResBlock


class DINONewVq(nn.Module):
    def __init__(self, cfg: dict, cfg_loss: dict):
        super().__init__()
        self.cfg = cfg
        self.cfg_loss = cfg_loss

        self.extractor = DinoFeaturizer(cfg["pretrained"])
        self.feat_dim = self.extractor.n_feats  # 384
        self.hidden_dim = cfg.get("hidden_dim", self.feat_dim)

        # -------- semantic-encoder -------- #
        num_enc_blocks = cfg["enc_num_blocks"]
        semantic_enc_proj = []
        for i in range(num_enc_blocks):
            semantic_enc_proj.append(EncResBlock(self.feat_dim if (i == 0) else self.hidden_dim, self.hidden_dim))
        self.enc_proj = nn.Sequential(*semantic_enc_proj)

        # -------- vq -------- #
        vq_num_codebooks = cfg["vq"]["num_codebooks"]
        vq_embed_dims = cfg["vq"]["embed_dims"]
        assert len(vq_num_codebooks) == len(vq_embed_dims)
        self.num_vq = len(vq_num_codebooks)
        self.beta = cfg["vq"]["beta"]
        self.vq_type = cfg["vq"]["vq_type"]
        self.use_weighted_sum = cfg["vq"].get("use_weighted_sum", False)
        self.need_initialized = cfg["vq"].get("need_initialized", False)

        self.num_pq = cfg["vq"].get("num_pq", 1)

        if isinstance(self.num_pq, int):
            self.num_pq = [self.num_pq] * self.num_vq

        if self.vq_type == "ema":
            raise ValueError("Not implemented")
            # vq_kwargs["decay"] = cfg["vq"]["decay"]
            # vq_kwargs["eps"] = cfg["vq"]["eps"]
            # vq_blocks = [
            #     EMAVectorQuantizer(vq_num_codebooks[i], vq_embed_dims[i], **vq_kwargs) if (self.num_pq == 1) else
            #     ProductQuantizerWrapper(self.num_pq[i], vq_num_codebooks[i], vq_embed_dims[i], **vq_kwargs,
            #                             quantizer_cls=EMAVectorQuantizer)
            #     for i in range(self.num_vq)
            # ]
        elif self.vq_type == "param":
            self.codebook = Codebook(num_codebook_vectors=vq_num_codebooks[0], latent_dim=vq_embed_dims[0],
                                     beta=self.beta)
        else:
            raise ValueError(f"Unsupported vq type {self.vq_type}.")

        # -------- semantic-decoder -------- #
        num_dec_blocks = cfg["dec_num_blocks"]
        dec_proj = []
        for i in range(num_dec_blocks):
            dec_proj.append(
                DecResBlock(self.hidden_dim, self.feat_dim if (i == num_dec_blocks - 1) else self.hidden_dim))
        self.dec_proj = nn.Sequential(*dec_proj)

    def forward(self, img: torch.Tensor
                ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        # # photometric aug
        # img_aug_1 = img
        # img_aug_2 = self._photo_aug(img)
        #
        # img = torch.cat([img_aug_1, img_aug_2], dim=0)  # (2b, 3, h, w)

        dino_feat = self.extractor(img)  # (b, 384, 28, 28)
        feat = self.enc_proj(dino_feat)  # (b, 384, 28, 28)

        output = dict()

        quantized_feat, vq_loss = self.codebook(feat)

        recon = self.dec_proj(quantized_feat)  # (2b, 384, 28, 28)
        recon_loss = F.mse_loss(recon, dino_feat)

        output["recon-loss"] = recon_loss
        output["vq-loss"] = vq_loss
        # # contrastive loss
        # top_dis_prob_1, top_dis_prob_2 = torch.chunk(vq_top_dis_prob, chunks=2, dim=0)  # (2bhw, K) -> (2, bhw, K)
        # output["contra-loss-pos"] = self.jsd(top_dis_prob_1, top_dis_prob_2)
        #
        # bottom_dis_prob_1, bottom_dis_prob_2 = torch.chunk(vq_bottom_dis_prob, chunks=2, dim=0)
        # output["contra-loss-neg"] = self.jsd(bottom_dis_prob_1, bottom_dis_prob_2)

        # split half
        # feat = torch.chunk(feat, chunks=2, dim=0)[0]
        # feat_vqs = [torch.chunk(vq_i, chunks=2, dim=0)[0] for vq_i in feat_vqs]
        return feat, quantized_feat, output


class Codebook(nn.Module):
    def __init__(self, num_codebook_vectors, latent_dim, beta=0.25):
        super(Codebook, self).__init__()
        """
        embedding: (num_vq, embed_dim)
        beta: the factor for vq_loss
        input: (b, embed_dim, h, w)
        output: (b, embed_dim, h, w)
        """
        # self.num_codebook_vectors = args.num_codebook_vectors
        self.latent_dim = latent_dim
        self.beta = beta

        self.embedding = nn.Embedding(num_codebook_vectors, latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_codebook_vectors, 1.0 / num_codebook_vectors)

    def forward(self, z : torch.Tensor):
        z = z.permute(0, 2, 3, 1).contiguous()  # (b, d, h, w) -> (b, h, w, d)
        z_flattened = z.view(-1, self.latent_dim)  # (bhw, d)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * (torch.matmul(z_flattened, self.embedding.weight.t()))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        q_loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, q_loss
