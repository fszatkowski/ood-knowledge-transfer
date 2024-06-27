from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
import wandb


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)


def save_hyperparams_to_wandb(args: Dict):
    for key, value in args.items():
        setattr(wandb.config, key, value)


class SaveEvery(pl.Callback):
    def __init__(self, every, save_dir):
        self.every = every
        self.save_dir = save_dir

    def on_train_epoch_end(self, trainer: pl.Trainer, _):
        """Check if we should save a checkpoint after every train epoch"""
        epoch = trainer.current_epoch
        if epoch % self.every == 0:
            ckpt_path = f"{self.save_dir}/ckpt_{epoch}.ckpt"
            trainer.save_checkpoint(ckpt_path)
