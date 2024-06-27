#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

set -e

eval "$(conda shell.bash hook)"
conda activate sid


python src/main.py \
  --cfg_path configs/distill/cifar10_cifar10_resnet18.yaml \
  seed=${seed} \
  teacher_ckpt=/data/fszatkowski/sid/pretrained/cifar10/resnet18/last.ckpt
  save_dir=/data/fszatkowski/sid/distill/cifar10_cifar10/resnet18
