import argparse
import os

import torch

from dotenv import load_dotenv
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger

import utils
from distiller import ImgDistill
from data_utils import prepare_data

parser = argparse.ArgumentParser(description="Knowledge Distillation From a Single Image.")

parser.add_argument("--gpus", default=-1, type=int, help="how many gpus to use")
parser.add_argument("--resume", action='store_true', help="if set will try to resume the run")

# Teacher settings
parser.add_argument("--teacher_arch", default="resnet18", type=str, help="arch for teacher")
parser.add_argument("--use_timm", action="store_true", help="use strong-aug trained timm models?")
parser.add_argument("--teacher_ckpt", default="", type=str, help="ckpt to load teacher. not needed for IN-1k")

# Student
parser.add_argument("--student_arch", default="resnet50", type=str, help="arch for student")
parser.add_argument("--temperature", default=8, type=float, help="temperature logits are divided by")

# LSH
parser.add_argument("--n_rvectors", default=[12, 13, 14], nargs="+", type=int, help="sizes of buckets")
parser.add_argument("--n_projections", default=20, type=int, help="number of projections for lsh averaging")

# Training settings
parser.add_argument("--lr_schedule", action="store_true", help="lr_schedule")
parser.add_argument("--milestones", default=[100, 150], nargs="*", type=int, help="lr schedule (drop lr by 5x)")
parser.add_argument("--lr_decay", default=0.5, type=float, help="lr decay for scheduler")
parser.add_argument("--epochs", default=200, type=int, help="number of total epochs to run")
parser.add_argument("--batch_size", default=512, type=int, help="batch size per GPU")

# Optimizer
parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
parser.add_argument("--wd", default=5e-4, type=float, help="weight decay")

# data
# parser.add_argument("--traindir", default="/tmp/train/", type=str, help="folder with folder(s) of training imgs")
# parser.add_argument("--testdir", default="/datasets/ILSVRC12/val/", type=str, help="folder with folder(s) of test imgs")
parser.add_argument("--dst_dataset", default="cifar100", type=str, help="Dataset to use for distillation")
parser.add_argument("--tgt_dataset", default="cifar100", type=str, help="Dataset to use for evaluation")
parser.add_argument('--limit_val_batches', default=1., type=float, help='fraction of validation batches to use')

# saving etc.
parser.add_argument("--save_dir", default="./output/", type=str, help="saving dir")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--save_every", default=10, type=int, help="save every n epochs")
parser.add_argument("--eval_every", default=1, type=int, help="save every n epochs")
parser.add_argument("--validate", default="", type=str, help="val only")

if __name__ == "__main__":
    args = parser.parse_args()

    load_dotenv()

    dst_loader, tgt_loader, num_classes = prepare_data(args)

    # setup logging and saving dirs
    ckpt_path = os.path.join(args.save_dir)
    wandb_logger = WandbLogger()
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        monitor="val/acc_top1",
        save_last=True, mode='max',
        filename=f"best"
    )
    last_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        save_last=True,
        filename=f"last"
    )

    # training module with teacher and student and optimizer
    distiller = ImgDistill(
        num_classes=num_classes,
        learning_rate=args.lr,
        weight_decay=args.wd,
        temperature=args.temperature,
        maxepochs=args.epochs,
        teacher_ckpt=args.teacher_ckpt,
        student_arch=args.student_arch,
        lr_schedule=args.lr_schedule,
        teacher_arch=args.teacher_arch,
        use_timm=args.use_timm,
        milestones=args.milestones,
        lr_decay=args.lr_decay,
        n_rvectors=args.n_rvectors,
        n_projections=args.n_projections,
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
        limit_val_batches=args.limit_val_batches
    )
    if args.validate != '':
        to_load = ckpt_path + f'/{args.validate}.ckpt'
        print("ckpt exists?:", os.path.isfile(to_load), flush=True)
        ckpt = torch.load(to_load, map_location='cpu')['state_dict']
        ckpt = {k.replace('student.', ''): v for k, v in ckpt.items() if 'student' in k}
        distiller.student.load_state_dict(ckpt)
        print("loading: ", to_load)
        trainer.test(distiller, dataloaders=tgt_loader)
    else:
        trainer.fit(distiller, dst_loader, tgt_loader)
        trainer.test(distiller, dataloaders=tgt_loader)
