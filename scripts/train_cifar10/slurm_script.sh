#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks-per-node=4   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

set -e

eval "$(conda shell.bash hook)"
conda activate ood_kd

batch_size=$1
lr=$2
wd=$3
decay=$4
seed=$5

python src/main.py \
  --cfg_path configs/train/cifar10_resnet18.yaml \
  batch_size_train=${batch_size} \
  learning_rate=${lr} \
  weight_decay=${wd} \
  lr_decay=${decay} \
  seed=${seed} \
  save_dir=pretrained/cifar10/resnet18_bs_${batch_size}_lr_${lr}_wd_${wd}_lr_decay_${decay}_seed_${seed}/last.ckpt \
  exp_name=train_cifar10_resnet18_bs_${batch_size}_lr_${lr}_wd_${wd}_lr_decay_${decay}