#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

set -e

eval "$(conda shell.bash hook)"
conda activate sid


for seed in 0 1 2; do
python src/main.py \
  --cfg_path configs/distill/cifar100_cifar100_resnet18.yaml \
  seed=${seed} \
  teacher_ckpt=/data/fszatkowski/sid/pretrained/cifar100/resnet18_${seed}/last.ckpt
  save_dir=/data/fszatkowski/sid/distill/cifar100_cifar100/resnet18_${seed}
done