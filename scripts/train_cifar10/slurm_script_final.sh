#!/bin/bash

set -e

for seed in 0 1 2;
  do sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/train_script_final.sh cifar10 $seed;
done