import argparse

from dotenv import load_dotenv
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger

import os

import torch
from torchvision.datasets import CIFAR100

import utils
import torchvision.transforms as transforms

from trainer import ImgTrain

CLASSES = {"in1k": 1000, "pets37": 37, "flowers102": 102, "stl10": 10, "places365": 365, 'in100': 100, 'cifar100': 100}

parser = argparse.ArgumentParser(description="Knowledge Distillation From a Single Image.")

parser.add_argument("--gpus", default=-1, type=int, help="how many gpus to use")
parser.add_argument("--resume", action='store_true', help="if set will try to resume the run")
parser.add_argument('--tags', nargs='+', default=None, type=str, help="tags for wandb")

# Teacher settings
parser.add_argument("--model_arch", default="resnet18_32", type=str, help="arch for teacher")

# Training settings
parser.add_argument("--lr_schedule", action="store_true", help="lr_schedule")
parser.add_argument("--milestones", default=[60, 120, 160], nargs="*", type=int, help="lr schedule (drop lr by 5x)")
parser.add_argument("--lr_decay", default=0.2, type=float, help="lr decay for scheduler")
parser.add_argument("--epochs", default=200, type=int, help="number of total epochs to run")
parser.add_argument("--batch_size", default=512, type=int, help="batch size per GPU")

# Optimizer
parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
parser.add_argument("--wd", default=5e-4, type=float, help="weight decay")

# data
parser.add_argument("--dataset", default="in1k", type=str, help="dataset name -- for saving and choosing num_classes")
parser.add_argument("--workers", default=8, type=int, help="number of workers")

# saving etc.
parser.add_argument("--save_dir", default="./output/", type=str, help="saving dir")
parser.add_argument("--save_every", default=10, type=int, help="save every n epochs")
parser.add_argument("--eval_every", default=1, type=int, help="save every n epochs")
parser.add_argument("--validate", default="", type=str, help="val only")


def prepare_data(args):
    if args.dataset == 'cifar100':
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        augmentations = [
            transforms.Pad(4),
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
        train_dataset = CIFAR100(root=os.environ['DATA_DIR'], download=True, train=True,
                                 transform=transforms.Compose(augmentations))

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
            transforms.ToTensor(),
            normalize
        ]
        # Define eval dataset
        val_dataset = CIFAR100(root=os.environ['DATA_DIR'], download=True, train=False,
                               transform=transforms.Compose(transform))
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            shuffle=False, persistent_workers=True)

        return train_loader, val_loader

    else:
        raise NotImplementedError()


if __name__ == "__main__":
    args = parser.parse_args()

    load_dotenv()

    train_loader, val_loader = prepare_data(args)

    # setup logging and saving dirs
    ckpt_path = os.path.join(args.save_dir)
    wandb_logger = WandbLogger(tags=args.tags)
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        monitor="val/acc_top1",
        save_last=True, mode='max',
        filename=f"best_{args.dataset}"
    )
    last_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        save_last=True,
        filename=f"last"
    )

    # training module with teacher and student and optimizer
    distiller = ImgTrain(
        num_classes=CLASSES[args.dataset],
        learning_rate=args.lr,
        weight_decay=args.wd,
        maxepochs=args.epochs,
        model_arch=args.model_arch,
        lr_schedule=args.lr_schedule,
        milestones=args.milestones,
        lr_decay=args.lr_decay,
    )

    # setup trainer
    if args.resume:
        resume = ckpt_path + '/last.ckpt' if os.path.isfile(ckpt_path + '/last.ckpt') else False
    else:
        resume = None
    trainer = Trainer(
        gpus=args.gpus, max_epochs=args.epochs,
        callbacks=[checkpoint_callback, last_callback, utils.SaveEvery(args.save_every, ckpt_path)],
        logger=wandb_logger,
        check_val_every_n_epoch=args.eval_every,
        progress_bar_refresh_rate=1, accelerator="ddp",
        plugins=[DDPPlugin(find_unused_parameters=False)],
        resume_from_checkpoint=resume,
    )
    if args.validate != '':
        to_load = ckpt_path + f'/{args.validate}.ckpt'
        print("ckpt exists?:", os.path.isfile(to_load), flush=True)
        ckpt = torch.load(to_load, map_location='cpu')['state_dict']
        ckpt = {k.replace('student.', ''): v for k, v in ckpt.items() if 'student' in k}
        distiller.model.load_state_dict(ckpt)
        print("loading: ", to_load)
        trainer.test(distiller, dataloaders=val_loader)
    else:
        trainer.fit(distiller, train_loader, val_loader)
        trainer.test(distiller, dataloaders=val_loader)
