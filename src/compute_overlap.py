import sys
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch_fidelity
from dotenv import load_dotenv
from einops import einops
from hydra import compose, initialize
from omegaconf import DictConfig
from torch import Tensor
from tqdm import tqdm

from data_utils import prepare_data
from main import set_seed
from model_utils import init_model


class FeatureHook:
    def __init__(self) -> None:
        self.features = None

    def capture(self, module, inp, out):
        self.features = out


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

    model = init_model(
        model_arch=cfg.model_arch,
        num_classes=cfg.num_classes,
        pretrained=cfg.pretrained,
        use_timm=cfg.use_timm,
        from_checkpoint=cfg.model_ckpt,
    )
    feature_layer = getattr(model, cfg.feature_layer)
    feature_hook = FeatureHook()
    feature_layer.register_forward_hook(feature_hook.capture)

    metrics = defaultdict(list)

    model.to(cfg.device)
    model.eval()

    with tqdm(
        total=cfg.n_passes * (len(train_loader_src) + len(train_loader_dst))
    ) as pbar:
        with torch.no_grad():
            for i in range(cfg.n_passes):
                with TemporaryDirectory() as tmpdir:
                    tmpdir = Path(tmpdir)
                    src_dir = tmpdir / "src"
                    dst_dir = tmpdir / "dst"
                    src_dir.mkdir(parents=True, exist_ok=True)
                    dst_dir.mkdir(parents=True, exist_ok=True)

                    for idx, (x, _) in enumerate(train_loader_src):
                        x = x.to(cfg.device, non_blocking=True)
                        model(x)
                        torch.save(feature_hook.features, src_dir / f"{idx}.pt")
                        pbar.update()
                    for idx, (x, _) in enumerate(train_loader_dst):
                        x = x.to(cfg.device, non_blocking=True)
                        model(x)
                        torch.save(feature_hook.features, dst_dir / f"{idx}.pt")
                        pbar.update()

                    # Load and merge features for bin computation
                    src_features = []
                    dst_features = []
                    for idx in range(len(train_loader_src)):
                        src_features.append(
                            torch.load(src_dir / f"{idx}.pt").detach().cpu()
                        )
                        dst_features.append(
                            torch.load(dst_dir / f"{idx}.pt").detach().cpu()
                        )
                    src_features = torch.cat(src_features, dim=0).squeeze()
                    dst_features = torch.cat(dst_features, dim=0).squeeze()
                    dataset_size_src = src_features.shape[0]
                    dataset_size_dst = dst_features.shape[0]
                    if dataset_size_src > dataset_size_dst:
                        print(
                            f"WARNING: dataset size src: {dataset_size_src} > dataset size dst: {dataset_size_dst}; "
                            f"downsampling to match the sizes"
                        )
                        src_features = src_features[:dataset_size_dst]
                    elif dataset_size_dst > dataset_size_src:
                        print(
                            f"WARNING: dataset size dst: {dataset_size_dst} > dataset size src: {dataset_size_src}; "
                            f"downsampling to match the sizes"
                        )
                        dst_features = dst_features[:dataset_size_src]
                    else:
                        continue

                    for proj_dim in cfg.lsh_proj_dims:
                        projections = [
                            torch.randn(src_features.shape[-1], proj_dim).type_as(
                                src_features
                            )
                            for _ in range(cfg.num_lsh_projs)
                        ]
                        projections = torch.stack(projections, dim=0).to(
                            src_features.device, non_blocking=True
                        )

                        mask = torch.ones((proj_dim,)) * 2
                        exponential = torch.arange(proj_dim)
                        multiply_matrix = torch.pow(mask, exponential).type(torch.int64)
                        multiply_matrix = multiply_matrix.to(
                            src_features.device, non_blocking=True
                        )

                        lsh_coverage_src = compute_lsh_coverage(
                            multiply_matrix, projections, src_features
                        )
                        lsh_coverage_dst = compute_lsh_coverage(
                            multiply_matrix, projections, dst_features
                        )
                        metrics[
                            f"hamming_distance_normalized_proj_dim_{proj_dim}"
                        ].append(
                            hamming_distance_normalized(
                                lsh_coverage_src, lsh_coverage_dst
                            )
                            .mean()
                            .item()
                        )

                        # TODO check if the features are normalized?

                        # TODO compute FID somehow?
                        # metrics = torch_fidelity.calculate_metrics(
                        #     input1=str(src_dir),
                        #     input2=str(dst_dir),
                        #     cuda=True,
                        #     isc=False,
                        #     fid=True,
                        #     kid=True,
                        #     prc=False,
                        #     verbose=True,
                        #     kid_subset_size=1000,
                        #     cache=False,
                        # )
                        # print(metrics)

    avg_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
    std_metrics = {k: np.std(v) for k, v in metrics.items()}
    print("Average metrics:\n", avg_metrics)
    print("Standard deviation of the metrics:\n", std_metrics)


def compute_lsh_coverage(multiply_matrix, projections, features):
    n_vectors = projections.shape[-1]
    features = einops.repeat(features, "b f -> n b f", n=cfg.num_lsh_projs)
    result: Tensor = torch.bmm(features, projections)  # [n, b, v]
    binary_hash = torch.gt(result, 0)  # [n, b, v]
    bin_ids = (binary_hash * multiply_matrix).sum(2).type(torch.int64)  # [n, b]
    covered_bins = torch.nn.functional.one_hot(bin_ids, num_classes=2**n_vectors).sum(
        dim=1
    )
    # Covered bins -> matrix of size num_lsh_projs, proj_dim**2, containing the number of times each bin was covered
    return covered_bins


def hamming_distance_normalized(coverage_src, coverage_dst):
    # Both coverages are expeted to have a batch first dimension, i.e. [num_lsh_projs, proj_dim**2]]
    return ((coverage_src > 0).float() != (coverage_dst > 0).float()).sum(
        1
    ) / coverage_src.shape[1]


if __name__ == "__main__":
    load_dotenv()
    cfg = parse_args()
    main(cfg)
