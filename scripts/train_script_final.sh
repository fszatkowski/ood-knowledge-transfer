#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks-per-node=4   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

set -e

eval "$(conda shell.bash hook)"
conda activate ood_kd

dataset=$1
seed=$2

python src/main.py \
  --cfg_path configs/train/${dataset}_resnet18.yaml \
  seed=${seed} \
  save_dir=pretrained/${dataset}/resnet18_seed_${seed} \
  exp_name=train_cifar100_${dataset}_resnet18