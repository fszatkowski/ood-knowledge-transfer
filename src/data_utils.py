import os
from typing import Callable, Tuple

import kornia.augmentation as kornia_transforms
import numpy as np
import torch
import torchvision.transforms as tv_transforms
from kornia import image_to_tensor
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import ConcatDataset, Subset
from torchvision.datasets import CIFAR10, CIFAR100
from medmnist import *


def prepare_data(
    train_dataset_name: str,
    test_dataset_name: str,
    cfg: DictConfig,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    dataset_name_list = ["cifar100", "cifar10", "cifar110", "pathmnist", "dermamnist"]

    if train_dataset_name in dataset_name_list:
        if cfg.use_kornia:
            augs_train = KorniaToTensor()
        else:
            augs_train = [
                tv_transforms.RandomResizedCrop(32, scale=(0.08, 1.0)),
                tv_transforms.RandomApply(
                    [tv_transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)], p=0.8
                ),
                tv_transforms.RandomGrayscale(p=0.2),
                tv_transforms.RandomSolarize(threshold=128, p=0.2),
                tv_transforms.RandomHorizontalFlip(),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
            augs_train = tv_transforms.Compose(augs_train)
        train_dataset = get_dataset(
            train_dataset_name, transform=augs_train, train=True
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.batch_size_train,
            num_workers=cfg.workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )
    else:
        raise NotImplementedError()

    if test_dataset_name in dataset_name_list:
        if cfg.use_kornia:
            augs_test = KorniaToTensor()
        else:
            augs_test = [
                tv_transforms.Resize(36),
                tv_transforms.CenterCrop(32),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
            augs_test = tv_transforms.Compose(augs_test)

        test_dataset = get_dataset(test_dataset_name, transform=augs_test, train=False)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=cfg.batch_size_eval,
            num_workers=cfg.workers,
            pin_memory=True,
            shuffle=False,
        )
    else:
        raise NotImplementedError()

    return train_loader, test_loader


def get_kornia_augs(dataset_name: str) -> Tuple[Callable, Callable]:
    # Returns kornia augmentations for a given dataset
    dataset_name_list = ["cifar100", "cifar10", "cifar110", "pathmnist", "dermamnist"]
    if dataset_name in dataset_name_list:
        augs_train = [
            kornia_transforms.RandomResizedCrop((32, 32), scale=(0.08, 1.0)),
            kornia_transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4, p=0.8
            ),
            kornia_transforms.RandomGrayscale(p=0.2),
            kornia_transforms.RandomSolarize(p=0.2),
            kornia_transforms.RandomHorizontalFlip(),
            kornia_transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            ),
        ]
        augs_test = [
            kornia_transforms.Resize(36),
            kornia_transforms.CenterCrop(32),
            kornia_transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            ),
        ]
        return kornia_transforms.AugmentationSequential(
            *augs_train
        ), kornia_transforms.AugmentationSequential(*augs_test)


def get_dataset(dataset_name: str, transform=None, train: bool = False):
    if dataset_name == "pathmnist":
        dataset = PathMNIST(
            root=os.environ["DATA_DIR"],
            download=True,
            size=64,
            split="train" if train else "test",
            transform=transform,
        )
    elif dataset_name == "dermamnist":
        dataset = DermaMNIST(
            root=os.environ["DATA_DIR"],
            download=True,
            size=64,
            split="train" if train else "test",
            transform=transform,
        )
    elif dataset_name == "cifar100":
        dataset = CIFAR100(
            root=os.environ["DATA_DIR"], download=True, train=train, transform=transform
        )
    elif dataset_name == "cifar10":
        dataset = CIFAR10(
            root=os.environ["DATA_DIR"], download=True, train=train, transform=transform
        )
    elif dataset_name == "cifar110":
        if train:
            ds1 = CIFAR100(
                root=os.environ["DATA_DIR"],
                download=True,
                train=True,
                transform=transform,
            )
            ds2 = CIFAR10(
                root=os.environ["DATA_DIR"],
                download=True,
                train=True,
                transform=transform,
            )
            ds1 = Subset(ds1, get_subset_of_indices(ds1, ratio=0.5))
            ds2 = Subset(ds2, get_subset_of_indices(ds2, ratio=0.5))
            dataset = ConcatDataset([ds1, ds2])
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    return dataset


def get_subset_of_indices(dataset, ratio: float = 0.5):
    unique_classes = set(dataset.targets)
    class_indices = {
        i: [idx for idx, cls in enumerate(dataset.targets) if cls == i]
        for i in range(len(unique_classes))
    }
    first_half_indices = [
        idx
        for indices in class_indices.values()
        for idx in indices[: int(len(indices) * ratio)]
    ]
    return first_half_indices


class KorniaToTensor(torch.nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x) -> Tensor:
        x_tmp: np.ndarray = np.array(x)  # HxWxC
        x_out: Tensor = image_to_tensor(x_tmp, keepdim=True)  # CxHxW
        return x_out.float() / 255.0


def cutmix(x):
    with torch.no_grad():
        lam = np.random.beta(1.0, 1.0)
        rand_index = torch.randperm(x.size()[0]).cuda()
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    return x


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2
