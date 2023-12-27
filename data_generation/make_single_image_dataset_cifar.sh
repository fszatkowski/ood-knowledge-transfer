#!/bin/bash

set -e

for num_imgs in 100 500 1000 5000 10000 50000; do
  python data_generation/make_single_img_dataset.py \
    --img_size 40 \
    --batch_size 25 \
    --num_imgs ${num_imgs} \
    --threads 4 \
    --imgpath data_generation/images/ameyoko.jpg \
    --targetpath /data/fszatkowski/sid/cifar/ameyoko/${num_imgs}/

  python data_generation/make_single_img_dataset.py \
    --img_size 40 \
    --batch_size 25 \
    --num_imgs ${num_imgs} \
    --threads 4 \
    --imgpath data_generation/images/img_a.jpg \
    --targetpath /data/fszatkowski/sid/cifar/animals/${num_imgs}/

  python data_generation/make_single_img_dataset.py \
    --img_size 40 \
    --batch_size 25 \
    --num_imgs ${num_imgs} \
    --threads 4 \
    --imgpath data_generation/images/sf.png \
    --targetpath /data/fszatkowski/sid/cifar/sf/${num_imgs}/

  python data_generation/make_single_img_dataset.py \
    --img_size 40 \
    --batch_size 25 \
    --num_imgs ${num_imgs} \
    --threads 4 \
    --imgpath data_generation/images/11_octaves.png \
    --targetpath /data/fszatkowski/sid/cifar/octaves/${num_imgs}/

  python data_generation/make_single_img_dataset.py \
    --img_size 40 \
    --batch_size 25 \
    --num_imgs ${num_imgs} \
    --threads 4 \
    --imgpath data_generation/images/hubble.jpg \
    --targetpath /data/fszatkowski/sid/cifar/hubble/${num_imgs}/
done

python data_generation/combine_datasets.py \
  --data_dir /data/fszatkowski/sid/cifar \
  --num_samples_src 10000
