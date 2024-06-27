import os
import sys
from logging import SaveEvery, save_hyperparams_to_wandb, set_seed

from dotenv import load_dotenv
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data_utils import prepare_data
from main_module import MainModule


def parse_args():
    args = sys.argv
    assert len(args) >= 3, "You must provide the config path"
    assert args[1] == "--cfg_path", "You must provide the config path"

    cfg_path = args[2]
    overrides = args[3:]

    config_dir = "/".join(cfg_path.split("/")[:-1])
    cfg_name = cfg_path.split("/")[-1].split(".")[0]
    with initialize(
        version_base=None, config_path=f"../{config_dir}", job_name="dynamic_config"
    ):
        cfg = compose(config_name=cfg_name, overrides=overrides)
        return cfg


def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    train_loader, test_loader = prepare_data(cfg)
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.save_dir, monitor="val/acc_top1", mode="max", filename="best"
    )
    last_callback = ModelCheckpoint(
        dirpath=cfg.save_dir, save_last=True, filename="last"
    )
    lr_callback = LearningRateMonitor(logging_interval="epoch")
    save_callback = SaveEvery(cfg.save_every, cfg.save_dir)
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
    main(cfg)
