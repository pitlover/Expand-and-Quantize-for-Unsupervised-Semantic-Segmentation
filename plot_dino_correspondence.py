import os
from os.path import join
from model.dino.dino_featurizer import DinoFeaturizer
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib.colors import ListedColormap
from utils.config_utils import prepare_config
from torchvision import transforms as T
from PIL import Image
from data.dataset_aug import UnSegDataset


def _remove_axes(ax):
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_xticks([])
    ax.set_yticks([])


def remove_axes(axes):
    if len(axes.shape) == 2:
        for ax1 in axes:
            for ax in ax1:
                _remove_axes(ax)
    else:
        for ax in axes:
            _remove_axes(ax)


def prep_for_plot(img, rescale=True, resize=None):
    if resize is not None:
        img = F.interpolate(img.unsqueeze(0), resize, mode="bilinear")
    else:
        img = img.unsqueeze(0)

    plot_img = unnorm(img).squeeze(0).cpu().permute(1, 2, 0)
    if rescale:
        plot_img = (plot_img - plot_img.min()) / (plot_img.max() - plot_img.min())
    return plot_img


def sample(t: torch.Tensor, coords: torch.Tensor):
    return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)


def plot_heatmap(ax, image, heatmap, cmap="bwr", color=False, plot_img=True, symmetric=True):
    vmax = np.abs(heatmap).max()
    if not color:
        bw = np.dot(np.array(image)[..., :3], [0.2989, 0.5870, 0.1140])
        image = np.ones_like(image) * np.expand_dims(bw, -1)

    if symmetric:
        kwargs = dict(vmax=vmax, vmin=-vmax)
    else:
        kwargs = {}

    if plot_img:
        return [
            ax.imshow(image),
            ax.imshow(heatmap, alpha=.5, cmap=cmap, **kwargs),
        ]
    else:
        return [ax.imshow(heatmap, alpha=.5, cmap=cmap, **kwargs)]


def get_heatmaps(net, img, img_pos, query_points):
    feats1 = net(img.cuda())
    feats2 = net(img_pos.cuda())

    sfeats1 = sample(feats1, query_points)

    attn_intra = torch.einsum("nchw,ncij->nhwij", F.normalize(sfeats1, dim=1), F.normalize(feats1, dim=1))
    attn_intra -= attn_intra.mean([3, 4], keepdims=True)
    attn_intra = attn_intra.clamp(0).squeeze(0)

    attn_inter = torch.einsum("nchw,ncij->nhwij", F.normalize(sfeats1, dim=1), F.normalize(feats2, dim=1))
    attn_inter -= attn_inter.mean([3, 4], keepdims=True)
    attn_inter = attn_inter.clamp(0).squeeze(0)

    heatmap_intra = F.interpolate(
        attn_intra, img.shape[2:], mode="bilinear", align_corners=True).squeeze(0).detach().cpu()
    heatmap_inter = F.interpolate(
        attn_inter, img_pos.shape[2:], mode="bilinear", align_corners=True).squeeze(0).detach().cpu()

    return heatmap_intra, heatmap_inter


class ToTargetTensor(object):
    def __call__(self, target):
        return torch.as_tensor(np.array(target), dtype=torch.int64).unsqueeze(0)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2


normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def get_transform(res, is_label, crop_type):
    if crop_type == "center":
        cropper = T.CenterCrop(res)
    elif crop_type == "random":
        cropper = T.RandomCrop(res)
    elif crop_type is None or crop_type == "none":
        cropper = T.Lambda(lambda x: x)
        res = (res, res)
    else:
        raise ValueError("Unknown Cropper {}".format(crop_type))
    if is_label:
        return T.Compose([T.Resize(res, Image.NEAREST),
                          cropper,
                          ToTargetTensor()])
    else:
        return T.Compose([T.Resize(res, Image.NEAREST),
                          cropper,
                          T.ToTensor(),
                          normalize])


def my_app(cfg: dict) -> None:
    pytorch_data_dir = '../Datasets/pascal'
    seed_everything(seed=0, workers=True)
    high_res = 512

    transform = get_transform(high_res, False, "center")
    use_loader = True

    if use_loader:
        dataset = UnSegDataset(
            mode="train",
            data_dir=cfg["data_dir"],
            dataset_name=cfg["dataset_name"],
            model_type=cfg["model"]["pretrained"]["model_type"],
            crop_type="none",
            transform=transform,
            target_transform=get_transform(high_res, True, "center"),
            num_neighbors=2,
            pos_images=True,
            pos_labels=True,
        )

        loader = DataLoader(dataset, 16, shuffle=True, num_workers=8)

    net = DinoFeaturizer(cfg["model"]["pretrained"])
    net = net.cuda()

    for batch_val in loader:
        batch = batch_val
        break

    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
    cmaps = [
        ListedColormap([(1, 0, 0, i / 255) for i in range(255)]),
        ListedColormap([(0, 1, 0, i / 255) for i in range(255)]),
        ListedColormap([(0, 0, 1, i / 255) for i in range(255)]),
        ListedColormap([(1, 1, 0, i / 255) for i in range(255)])
    ]

    with torch.no_grad():

        img_num = 6
        query_points = torch.tensor(
            [
                [-.1, 0.0],
                [.5, .8],
                [-.7, -.7],
            ]
        ).reshape(1, 3, 1, 2).cuda()

        img = batch["img"][img_num:img_num + 1]
        img_pos = batch["img_pos"][img_num:img_num + 1]

        plt.style.use('dark_background')
        fig, axes = plt.subplots(1, 3, figsize=(3 * 5, 1 * 5), dpi=100)
        remove_axes(axes)
        axes[0].set_title("Image and Query Points", fontsize=20)
        axes[1].set_title("Self Correspondence", fontsize=20)
        axes[2].set_title("KNN Correspondence", fontsize=20)
        fig.tight_layout()

        heatmap_intra, heatmap_inter = get_heatmaps(net, img, img_pos, query_points)
        for point_num in range(query_points.shape[1]):
            point = ((query_points[0, point_num, 0] + 1) / 2 * high_res).cpu()
            img_point_h = point[0]
            img_point_w = point[1]

            plot_img = point_num == 0
            if plot_img:
                axes[0].imshow(prep_for_plot(img[0]))
            axes[0].scatter(img_point_h, img_point_w,
                            color=colors[point_num], marker="x", s=500, linewidths=5)

            plot_heatmap(axes[1], prep_for_plot(img[0]) * .8, heatmap_intra[point_num],
                         plot_img=plot_img, cmap=cmaps[point_num], symmetric=False)
            plot_heatmap(axes[2], prep_for_plot(img_pos[0]) * .8, heatmap_inter[point_num],
                         plot_img=plot_img, cmap=cmaps[point_num], symmetric=False)

        plt.savefig('./output/corr.png')


if __name__ == "__main__":
    args, config = prepare_config()
    my_app(config)
