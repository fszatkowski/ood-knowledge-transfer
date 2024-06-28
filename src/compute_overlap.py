import sys

from dotenv import load_dotenv
from hydra import compose, initialize
from omegaconf import DictConfig

from data_utils import prepare_data
from src.main import set_seed
from src.model_utils import init_model


class FeatureHook:
    # TODO
    pass


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
    train_loader_src, test_loader_src = prepare_data(
        train_dataset_name=cfg.src_dataset,
        test_dataset_name=cfg.src_dataset,
        cfg=cfg,
    )
    train_loader_dst, test_loader_dst = prepare_data(
        train_dataset_name=cfg.dst_dataset,
        test_dataset_name=cfg.dst_dataset,
        cfg=cfg,
    )

    model = init_model(model_arch=cfg.model_arch, from_checkpoint=cfg.ckpt_path)
    feature_layer = getattr(model, cfg.feature_layer)


if __name__ == "__main__":
    load_dotenv()
    cfg = parse_args()
    main(cfg)
