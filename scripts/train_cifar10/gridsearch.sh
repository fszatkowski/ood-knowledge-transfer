#!/bin/bash

set -e

batch_sizes=(128 256)
lrs=(0.1 0.01)
wds=(0.0005 0.0)
decays=(0.1 0.2)
seeds=(0 1 2)

for seed in "${seeds[@]}"; do
  for batch_size in "${batch_sizes[@]}"; do
    for lr in "${lrs[@]}"; do
      for wd in "${wds[@]}"; do
        for decay in "${decays[@]}"; do
          sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/train_cifar10/slurm_script.sh $batch_size $lr $wd $decay $seed
        done
      done
    done
  done
done

