#!/bin/bash

set -e

dataset_src=cifar100
dataset_tgt=cifar10
batch_size=128
lr=0.1

for seed in 0 1 2; do
  for temperature in 1 2 4 8 16; do
    for cutmix in True False; do
      sbatch --begin=now+3hours -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/distill_script_cosine.sh $dataset_src $dataset_tgt $batch_size $lr $temperature $cutmix $seed
      sbatch --begin=now+3hours -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/distill_script_multistep.sh $dataset_src $dataset_tgt $batch_size $lr $temperature $cutmix $seed
    done
  done
done