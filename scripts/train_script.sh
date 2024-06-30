#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks-per-node=4   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

set -e

eval "$(conda shell.bash hook)"
conda activate ood_kd

dataset=$1
batch_size=$2
lr=$3
wd=$4
decay=$5
seed=$6

python src/main.py \
  --cfg_path configs/train/${dataset}_resnet18.yaml \
  batch_size_train=${batch_size} \
  learning_rate=${lr} \
  weight_decay=${wd} \
  lr_decay=${decay} \
  seed=${seed} \
  save_dir=pretrained/${dataset}/resnet18_bs_${batch_size}_lr_${lr}_wd_${wd}_lr_decay_${decay}_seed_${seed} \
  exp_name=train_${dataset}_resnet18_bs_${batch_size}_lr_${lr}_wd_${wd}_lr_decay_${decay}