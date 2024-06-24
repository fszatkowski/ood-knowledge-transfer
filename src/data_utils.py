import os

import torch
from torch.utils.data import Subset, ConcatDataset
from torchvision.datasets import CIFAR100, CIFAR10

import utils
import torchvision.transforms as transforms

CLASSES = {"in1k": 1000, "pets37": 37, "flowers102": 102, "stl10": 10, "places365": 365, 'in100': 100, 'cifar100': 100,
           'cifar10': 10, 'cifar110': 110}


def prepare_data(args):
    # if args.dataset == 'in1k':
    #     # TODO imagenet code - prob will not work out of the box now
    #     # Define training augmentations
    #     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                      std=[0.229, 0.224, 0.225])
    #
    #     augmentations = [
    #         transforms.RandomResizedCrop(224, scale=(0.08, 1.), interpolation=3),
    #         transforms.RandomApply([
    #             transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)
    #         ], p=0.8),
    #         transforms.RandomGrayscale(p=0.2),
    #         transforms.RandomApply([utils.Solarize()], p=0.2),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize
    #     ]
    #
    #     # Define training dataset
    #     train_dataset = datasets.ImageFolder(
    #         args.traindir,
    #         transforms.Compose(augmentations))
    #     assert len(train_dataset) == args.n_samples, f'Expected {args.n_samples} samples, but got {len(train_dataset)}'
    #
    #     train_loader = torch.utils.data.DataLoader(
    #         train_dataset,
    #         batch_size=args.batch_size,
    #         num_workers=args.workers,
    #         pin_memory=True,
    #         shuffle=True,
    #         drop_last=True
    #     )
    #
    #     # Define eval augmentations
    #     transform = [
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize
    #     ]
    #     # Define eval dataset
    #     val_dataset = datasets.ImageFolder(
    #         args.testdir,
    #         transforms.Compose(transform))
    #     val_loader = torch.utils.data.DataLoader(
    #         val_dataset,
    #         batch_size=args.batch_size,
    #         num_workers=args.workers,
    #         pin_memory=True,
    #         shuffle=False, persistent_workers=True)
    #
    #     return train_loader, val_loader, CLASSES[args.dataset]

    if args.dst_dataset == 'cifar100' or args.dst_dataset == 'cifar10' or args.dst_dataset == 'cifar110':
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        augs_dst = [
            transforms.RandomResizedCrop(32, scale=(0.08, 1.), interpolation=3),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([utils.Solarize()], p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
        augs_dst = transforms.Compose(augs_dst)
        dst_set = get_dataset(args.dst_dataset, transform=augs_dst, train=True)
        dst_loader = torch.utils.data.DataLoader(
            dst_set,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True
        )
    else:
        raise NotImplementedError()

    if args.tgt_dataset == 'cifar100' or args.tgt_dataset == 'cifar10' or args.tgt_dataset == 'cifar110':
        # Define eval augmentations
        augs_tgt = [
            transforms.Resize(36),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            normalize
        ]
        augs_tgt = transforms.Compose(augs_tgt)
        tgt_set = get_dataset(args.tgt_dataset, transform=augs_tgt, train=False)
        tgt_loader = torch.utils.data.DataLoader(
            tgt_set,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            shuffle=False,
        )
        tgt_classes = CLASSES[args.tgt_dataset]
    else:
        raise NotImplementedError()

    return dst_loader, tgt_loader, tgt_classes


def get_dataset(dataset_name: str, transform=None, train: bool = False):
    if dataset_name == 'cifar100':
        dataset = CIFAR100(root=os.environ['DATA_DIR'], download=True, train=train, transform=transform)
    elif dataset_name == 'cifar10':
        dataset = CIFAR10(root=os.environ['DATA_DIR'], download=True, train=train, transform=transform)
    elif dataset_name == 'cifar110':
        if train:
            ds1 = CIFAR100(root=os.environ['DATA_DIR'], download=True, train=True, transform=transform)
            ds2 = CIFAR10(root=os.environ['DATA_DIR'], download=True, train=True, transform=transform)
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
    class_indices = {i: [idx for idx, cls in enumerate(dataset.targets) if cls == i] for i in range(len(unique_classes))}
    first_half_indices = [idx for indices in class_indices.values() for idx in indices[:int(len(indices) * ratio)]]
    return first_half_indices
