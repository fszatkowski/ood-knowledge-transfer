#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

set -e

eval "$(conda shell.bash hook)"
conda activate sid


for seed in 0 1 2; do
python src/main.py \
  --cfg_path configs/train/cifar10_resnet18.yaml \
  seed=${seed} \
  save_dir=/data/fszatkowski/sid/pretrained/cifar10/resnet18_${seed}
done