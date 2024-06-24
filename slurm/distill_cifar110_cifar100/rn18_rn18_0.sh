#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

set -e

eval "$(conda shell.bash hook)"
conda activate sid

# Setting
dst_dataset=cifar110
tgt_dataset=cifar100
teacher_arch=resnet18_32
student_arch=resnet18_32

# distillation args
temperature=8
batch_size=512
lr=0.01
epochs=200
milestones="100 150"
lr_decay=0.5
save_path_string=tmp_${temperature}_batch_size_${batch_size}_lr_${lr}

python src/distill.py \
  --teacher_arch ${teacher_arch} \
  --teacher_ckpt /data/fszatkowski/sid/pretrained_models/cifar100/resnet18/last.ckpt \
  --student_arch ${student_arch} \
  --dst_dataset ${dst_dataset} \
  --tgt_dataset ${tgt_dataset} \
  --gpus 1 \
  --batch_size ${batch_size} \
  --epochs ${epochs} \
  --lr ${lr} \
  --milestones ${milestones} \
  --lr_decay ${lr_decay} \
  --temperature ${temperature} \
  --lr_schedule \
  --save_dir /data/fszatkowski/sid/kd_test/dst_${dst_dataset}_${teacher_arch}_tgt_${tgt_dataset}_${student_arch}/${save_path_string}

