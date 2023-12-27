import os

import torch
from torchvision.datasets import CIFAR100

import utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def prepare_data(args):
    if args.dataset == 'in1k':
        # Define training augmentations
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        augmentations = [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.), interpolation=3),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([utils.Solarize()], p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

        # Define training dataset
        train_dataset = datasets.ImageFolder(
            args.traindir,
            transforms.Compose(augmentations))
        assert len(train_dataset) == args.n_samples, f'Expected {args.n_samples} samples, but got {len(train_dataset)}'

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True
        )

        # Define eval augmentations
        transform = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]
        # Define eval dataset
        val_dataset = datasets.ImageFolder(
            args.testdir,
            transforms.Compose(transform))
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            shuffle=False, persistent_workers=True)

        return train_loader, val_loader
    elif args.dataset == 'cifar100':
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        augmentations = [
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

        # Define training dataset
        train_dataset = datasets.ImageFolder(
            args.traindir,
            transforms.Compose(augmentations))
        assert len(train_dataset) == args.n_samples, f'Expected {args.n_samples} samples, but got {len(train_dataset)}'

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True
        )

        # Define eval augmentations
        transform = [
            transforms.Resize(36),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            normalize
        ]
        # Define eval dataset
        val_dataset = CIFAR100(root=os.environ['DATA_DIR'], download=True, train=False, transform=transforms.Compose(transform))
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            shuffle=False, persistent_workers=True)

        return train_loader, val_loader

    else:
        raise NotImplementedError()