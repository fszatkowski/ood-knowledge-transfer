import sys

from hydra import compose, initialize


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
