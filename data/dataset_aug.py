from typing import Optional
import os
import random
from os.path import join
import numpy as np
import torch.multiprocessing
import torchvision.transforms as T  # noqa
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.cityscapes import Cityscapes
from torchvision.transforms.transforms import InterpolationMode

from .dataset_utils import coco_to_sparse, ToTargetTensor

__all__ = ["UnSegDataset"]


def get_transform(res: int, is_label: bool, crop_type: str, is_aug: bool = False):
    if crop_type == "center":
        cropper = T.CenterCrop(res)
    elif crop_type == "random":
        cropper = T.RandomCrop(res)
    elif crop_type is None:
        cropper = T.Lambda(lambda x: x)
        res = (res, res)
    else:
        raise ValueError("Unknown Cropper {}".format(crop_type))

    if is_label:
        return T.Compose([
            T.Resize(res, InterpolationMode.NEAREST),
            cropper,
            ToTargetTensor()
        ])
    elif is_aug:
        return T.Compose([
            T.Resize(res, InterpolationMode.NEAREST),
            cropper,
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.RandomGrayscale(0.2),
            T.RandomApply([T.GaussianBlur(3, 3)]),
            T.ToTensor(),  # (h, w, 3) [0 255] -> (3, h, w) [0, 1]
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize(res, InterpolationMode.NEAREST),
            cropper,
            T.ToTensor(),  # (h, w, 3) [0 255] -> (3, h, w) [0, 1]
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


class CocoSeg(Dataset):
    def __init__(self,
                 mode: str,
                 data_dir: str,
                 transform=None,
                 target_transform=None,
                 aug_transform=None,
                 coarse_labels: bool = False,
                 exclude_things: bool = False,
                 subset: Optional[int] = None,
                 ):
        super().__init__()

        assert mode in ("train", "val", "train+val")
        split_dirs = {
            "train": ["train2017"],
            "val": ["val2017"],
            "train+val": ["train2017", "val2017"]
        }
        self.mode = mode
        self.data_dir = data_dir

        if (transform is None) or (target_transform is None):
            raise ValueError("Transform is None")

        self.transform = transform
        self.target_transform = target_transform
        self.aug_transform = aug_transform
        self.coarse_labels = coarse_labels  # only True for cocostuff3
        self.exclude_things = exclude_things
        self.subset = subset

        if self.subset is None:
            self.image_list = "Coco164kFull_Stuff_Coarse.txt"
        elif self.subset == 6:  # IIC Coarse
            self.image_list = "Coco164kFew_Stuff_6.txt"
        elif self.subset == 7:  # IIC Fine
            self.image_list = "Coco164kFull_Stuff_Coarse_7.txt"

        self.image_files = []
        self.label_files = []
        for split_dir in split_dirs[self.mode]:
            with open(join(self.data_dir, "curated", split_dir, self.image_list), "r") as f:
                img_ids = [fn.rstrip() for fn in f.readlines()]
                for img_id in img_ids:
                    self.image_files.append(join(self.data_dir, "images", split_dir, img_id + ".jpg"))
                    self.label_files.append(join(self.data_dir, "annotations", split_dir, img_id + ".png"))

        self.fine_to_coarse = coco_to_sparse()
        # self._label_names = ["ground-stuff", "plant-stuff", "sky-stuff"]

        self.cocostuff3_coarse_classes = [23, 22, 21]
        self.first_stuff_index = 12

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int):
        """COCO dataset get item
        :param index:       int
        :return:            img (3, h, w)
                            label (h, w): LongTensor label indices for each pixel
                            label_mask (h, w): BoolTensor (T: valid, F: invalid)
                            image_path: str
        """
        image_path = self.image_files[index]
        label_path = self.label_files[index]

        img = self.transform(Image.open(image_path).convert("RGB"))  # (3, h, w)
        label = self.target_transform(Image.open(label_path)).squeeze(0)  # (1, h, w) -> (h, w)
        img_aug = self.aug_transform(Image.open(image_path).convert("RGB"))  # (3, h, w)

        label[label == 255] = -1  # to be consistent with 10k (coco-10k??)
        coarse_label = torch.zeros_like(label)
        for fine, coarse in self.fine_to_coarse.items():
            coarse_label[label == fine] = coarse
        coarse_label[label == -1] = -1

        if self.coarse_labels:
            coarser_labels = -torch.ones_like(label)
            for i, c in enumerate(self.cocostuff3_coarse_classes):
                coarser_labels[coarse_label == c] = i
            return img, img_aug, coarser_labels, (coarser_labels >= 0), image_path
        else:
            if self.exclude_things:
                return img, img_aug, (coarse_label - self.first_stuff_index), \
                       (coarse_label >= self.first_stuff_index), image_path
            else:
                return img, img_aug, coarse_label, (coarse_label >= 0), image_path


class CityscapesSeg(Dataset):
    def __init__(self,
                 mode: str,
                 data_dir: str,
                 transform=None,
                 target_transform=None,
                 ):
        super().__init__()

        assert mode in ("train", "val", "train_extra")
        self.mode = mode
        self.data_dir = data_dir

        if mode == "train_extra":
            inner_mode = "coarse"
        else:  # "train", "val"
            inner_mode = "fine"

        self.inner_dataset = Cityscapes(
            data_dir, split=mode, mode=inner_mode,
            target_type="semantic",
            transform=None,
            target_transform=None
        )

        self.transform = transform
        self.target_transform = target_transform
        self.first_non_void = 7

    def __len__(self) -> int:
        return len(self.inner_dataset)

    def __getitem__(self, index: int):
        """CityScape get item
        :param index:       int
        :return:            img (3, h, w)
                            label (h, w): LongTensor label indices for each pixel
                            label_mask (h, w): BoolTensor (T: valid, F: invalid)
                            image_path: str
        """
        image, target = self.inner_dataset[index]
        image_path = self.inner_dataset.images[index]

        if self.transform is not None:
            image = self.transform(image)
            target = self.target_transform(target).squeeze(0)

            target = (target - self.first_non_void)
            target[target < 0] = -1
            mask = (target == -1)  # TODO is this valid or non-valid??
            return image, target, mask, image_path
        else:
            mask = torch.zeros_like(target, dtype=torch.bool)  # TODO filled with False
            return image, target, mask, image_path


class CroppedDataset(Dataset):
    def __init__(self,
                 mode: str,
                 data_dir: str,
                 dataset_name: str,
                 crop_type: str = "five",
                 crop_ratio: float = 0.5,
                 transform=None,
                 target_transform=None,
                 aug_transform=None,
                 ):
        super().__init__()

        self.mode = mode
        self.dataset_name = dataset_name

        self.data_dir = join(data_dir, "cropped", f"{dataset_name}_{crop_type}_crop_{crop_ratio}")

        if (transform is None) or (target_transform is None):
            raise ValueError("Transform is None")

        self.transform = transform
        self.target_transform = target_transform
        self.aug_transform = aug_transform

        self.img_dir = join(self.data_dir, "img", self.mode)
        self.label_dir = join(self.data_dir, "label", self.mode)

        self.num_images = len(os.listdir(self.img_dir))
        assert self.num_images == len(os.listdir(self.label_dir))

    def __len__(self) -> int:
        return self.num_images

    def __getitem__(self, index: int):
        """Cropped dataset get item
        :param index:       int
        :return:            img (3, h, w)
                            label (h, w): LongTensor label indices for each pixel
                            label_mask (h, w): BoolTensor (T: valid, F: invalid)
                            image_path: str
        """
        image_path = join(self.img_dir, "{}.jpg".format(index))
        label_path = join(self.label_dir, "{}.png".format(index))
        image_ = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)

        image = self.transform(image_)  # (3, 224, 224)
        label = self.target_transform(label).squeeze(0)  # (224, 224)
        img_aug = self.aug_transform(image_)

        label = (label - 1)
        mask = (label == -1)
        return image, img_aug, label, mask, image_path


class UnSegDataset(Dataset):
    def __init__(self,
                 mode: str,  # train, val
                 data_dir: str,
                 dataset_name: str,
                 model_type : str, # vit_small, vit_base
                 crop_type: Optional[str],  # 5-crop
                 crop_ratio: float = 0.5,
                 loader_crop_type: str = "center",  # center, random
                 res: int = 224,
                 pos_labels: bool = False,
                 pos_images: bool = False,
                 num_neighbors: int = 5
                 ):
        super().__init__()
        self.mode = mode
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.pos_labels = pos_labels
        self.pos_images = pos_images
        self.num_neighbors = num_neighbors

        if dataset_name == "cityscapes" and crop_type is None:
            self.n_classes = 27
            dataset_class = CityscapesSeg
            extra_args = dict()
        elif dataset_name == "cityscapes" and crop_type is not None:
            self.n_classes = 27
            dataset_class = CroppedDataset
            extra_args = dict(dataset_name="cityscapes", crop_type=crop_type, crop_ratio=crop_ratio)
        elif dataset_name == "cocostuff3":
            self.n_classes = 3
            dataset_class = CocoSeg
            extra_args = dict(coarse_labels=True, subset=6, exclude_things=True)
        elif dataset_name == "cocostuff15":
            self.n_classes = 15
            dataset_class = CocoSeg
            extra_args = dict(coarse_labels=False, subset=7, exclude_things=True)
        elif dataset_name == "cocostuff27" and crop_type is not None:
            # common training
            self.n_classes = 27
            dataset_class = CroppedDataset
            extra_args = dict(dataset_name="cocostuff27", crop_type=crop_type, crop_ratio=crop_ratio)
        elif dataset_name == "cocostuff27" and ((crop_type is None) or (crop_type == "none")):
            # common evaluation
            self.n_classes = 27
            dataset_class = CocoSeg
            extra_args = dict(coarse_labels=False, subset=None, exclude_things=False)
            if mode == "val":
                extra_args["subset"] = 7  # noqa
        else:
            raise ValueError("Unknown dataset: {}".format(dataset_name))

        transform = get_transform(res, is_label=False, crop_type=loader_crop_type)
        target_transform = get_transform(res, is_label=True, crop_type=loader_crop_type)
        aug_transform = get_transform(res, is_label=False, crop_type=loader_crop_type, is_aug=True)

        if "train" in mode:
            aug_geometric_transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResizedCrop(size=res, scale=(0.8, 1.0))
            ])
            aug_photometric_transform = T.Compose([
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                # T.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2), # simclr, swav style
                T.RandomGrayscale(0.2),
                T.RandomApply([T.GaussianBlur((5, 5))])
            ])
        else:
            aug_geometric_transform = aug_photometric_transform = None

        self.transform = transform
        self.target_transform = target_transform
        self.aug_transform = aug_transform
        self.aug_geometric_transform = aug_geometric_transform
        self.aug_photometric_transform = aug_photometric_transform

        self.dataset = dataset_class(
            data_dir=data_dir,
            mode=self.mode,
            transform=transform,
            target_transform=target_transform,
            aug_transform=aug_transform,
            **extra_args
        )

        feature_cache_file = join("../Datasets/cocostuff", "nns",
                                  f"nns_{model_type}_cocostuff27_{mode}_{crop_type}_224.npz")

        if self.pos_labels or self.pos_images:
            if not os.path.exists(feature_cache_file):
                raise ValueError("could not find nn file {} please run precompute_knns".format(feature_cache_file))
            else:
                loaded = np.load(feature_cache_file)
                self.nns = loaded["nns"]
            assert len(self.dataset) == self.nns.shape[0]

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _set_seed(seed):
        random.seed(seed)  # apply this seed to img transforms
        torch.manual_seed(seed)  # needed for torchvision 0.7

    def __getitem__(self, index: int):
        img, aug_img, label, mask, image_path = self.dataset[index]

        ret = {
            "index": index,
            "img": img,
            "aug_img": aug_img,
            "label": label,
            "mask": mask,
            "img_path": image_path
        }

        if self.pos_images or self.pos_labels:
            ind_pos = self.nns[index][torch.randint(low=1, high=self.num_neighbors + 1, size=[]).item()]
            img_pos, aug_img_pos, label_pos, mask_pos, image_path_pos = self.dataset[ind_pos]
            ret["index_pos"] = ind_pos
            ret["img_pos"] = img_pos
            ret["aug_img_pos"] = aug_img_pos
            ret["label_pos"] = label_pos
            ret["mask_pos"] = mask_pos
            ret["image_path_pos"] = image_path_pos

        return ret
