#!/bin/bash

conda activate sid

CUDA_VISIBLE_DEVICES=2 python src/main.py --cfg_path ./configs/train/pathmnist_resnet18.yaml
CUDA_VISIBLE_DEVICES=2 python src/main.py --cfg_path ./configs/train/medmnist_resnet18.yaml