import torch
import torch.nn.functional as F
import numpy as np
import os
from collections import defaultdict
from os.path import join
from data.dataset_utils import create_cityscapes_colormap, create_pascal_label_colormap
from PIL import Image


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2


unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def prep_for_plot(img, rescale=True, resize=None):
    if resize is not None:
        img = F.interpolate(img.unsqueeze(0), resize, mode="bilinear")
    else:
        img = img.unsqueeze(0)
    plot_img = unnorm(img).squeeze(0).squeeze(0).cpu().permute(1, 2, 0)
    if rescale:
        plot_img = (plot_img - plot_img.min()) / (plot_img.max() - plot_img.min())
    return plot_img


def visualization(save_dir: str, dataset_type: str, saved_data: defaultdict, cluster_metrics,
                  is_label: bool = False):
    os.makedirs(save_dir, exist_ok=True)
    if is_label:
        os.makedirs(join(save_dir, "label"), exist_ok=True)
    os.makedirs(join(save_dir, "cluster"), exist_ok=True)
    os.makedirs(join(save_dir, "linear"), exist_ok=True)

    if dataset_type.startswith("cityscapes"):
        label_cmap = create_cityscapes_colormap()
    else:
        label_cmap = create_pascal_label_colormap()

    for index in range(len(saved_data["img_path"])):
        file_name = str(saved_data["img_path"][index]).split("/")[-1].split(".")[0]

        if is_label:
            plot_label = label_cmap[saved_data["label"][index]].astype(np.uint8)
            Image.fromarray(plot_label).save(join(join(save_dir, "label", file_name + ".png")))

        plot_cluster = label_cmap[cluster_metrics.map_clusters(saved_data["cluster_preds"][index])].astype(np.uint8)
        Image.fromarray(plot_cluster).save(join(join(save_dir, "cluster", file_name + ".png")))

        plot_linear = label_cmap[saved_data["linear_preds"][index]].astype(np.uint8)
        Image.fromarray(plot_linear).save(join(join(save_dir, "linear", file_name + ".png")))
