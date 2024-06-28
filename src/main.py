import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from commandline_utils import parse_args
from data_utils import prepare_data
from main_module import MainModule


class SaveEvery(Callback):
    def __init__(self, every, save_dir):
        self.every = every
        self.save_dir = save_dir

    def on_train_epoch_end(self, trainer: Trainer, _):
        """Check if we should save a checkpoint after every train epoch"""
        epoch = trainer.current_epoch
        if epoch % self.every == 0:
            ckpt_path = f"{self.save_dir}/ckpt_{epoch}.ckpt"
            trainer.save_checkpoint(ckpt_path)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed)


def save_hyperparams_to_wandb(args: Dict):
    for key, value in args.items():
        setattr(wandb.config, key, value)


def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    train_loader, test_loader = prepare_data(
        train_dataset_name=cfg.train_dataset,
        test_dataset_name=cfg.test_dataset,
        cfg=cfg,
    )
    # Callbacks
    save_dir = Path(os.environ["RESULTS_DIR"]) / cfg.save_dir
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir, monitor="val/acc_top1", mode="max", filename="best"
    )
    last_callback = ModelCheckpoint(dirpath=save_dir, save_last=True, filename="last")
    lr_callback = LearningRateMonitor(logging_interval="epoch")
    save_callback = SaveEvery(cfg.save_every, save_dir)
    callbacks = [checkpoint_callback, last_callback, lr_callback, save_callback]

    if cfg.wandb:
        wandb_logger = WandbLogger(
            name=cfg.exp_name,
            project=os.environ["WANDB_PROJECT"],
            entity=os.environ["WANDB_ENTITY"],
            tags=cfg.tags,
        )
        save_hyperparams_to_wandb(OmegaConf.to_container(cfg))
        logger = wandb_logger
    else:
        logger = True

    trainer = Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        callbacks=callbacks,
        logger=logger,
        max_epochs=cfg.maxepochs,
        check_val_every_n_epoch=cfg.eval_every,
    )
    main_system = MainModule(cfg)
    if cfg.mode == "train" or cfg.mode == "distill":
        trainer.fit(main_system, train_loader, test_loader)
    elif cfg.mode == "eval":
        trainer.test(main_system, test_loader)
    else:
        raise NotImplementedError('"mode" must be one of ("train", "distill", "eval")')


if __name__ == "__main__":
    load_dotenv()
    cfg = parse_args()
    wandb.init(
        name=f"{cfg.mode}_testset_{cfg.test_dataset}_trainset_{cfg.train_dataset}"
    )
    main(cfg)
