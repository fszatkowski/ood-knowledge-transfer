#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks-per-node=4   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

set -e

eval "$(conda shell.bash hook)"
conda activate ood_kd

src_dataset=$1
tgt_dataset=$2

batch_size=$3
lr=$4
temperature=$5
cutmix=$6
seed=$7

python src/main.py \
  --cfg_path configs/distill/${src_dataset}_${tgt_dataset}_resnet18.yaml \
  batch_size_train=${batch_size} \
  learning_rate=${lr} \
  seed=${seed} \
  temperature=${temperature} \
  cutmix=${cutmix} \
  scheduler=cosine \
  teacher_ckpt=pretrained/${src_dataset}/resnet18_seed_${seed}/last.ckpt \
  save_dir=distill/${src_dataset}_${tgt_dataset}/resnet18_cosine_bs_${batch_size}_lr_${lr}_temperature_${temperature}_cutmix_${cutmix}_seed_${seed} \
  exp_name=distill_${src_dataset}_${tgt_dataset}_resnet18_cosine_bs_${batch_size}_lr_${lr}_temperature_${temperature}_cutmix_${cutmix}

