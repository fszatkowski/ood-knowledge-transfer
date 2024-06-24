#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

set -e

eval "$(conda shell.bash hook)"
conda activate sid

MODEL_ARCH=resnet18_32
BATCH_SIZE=128

python src/train.py \
  --dataset cifar100 \
  --tags cifar100_pretraining \
  --model_arch ${MODEL_ARCH} \
  --gpus 1 \
  --batch_size ${BATCH_SIZE} \
  --epochs 200 \
  --lr_schedule \
  --save_dir /data/fszatkowski/sid/pretrained_models/cifar100/resnet18
