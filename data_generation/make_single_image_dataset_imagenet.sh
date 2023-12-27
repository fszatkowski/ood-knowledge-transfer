#!/bin/bash

set -e

NUM_IMGS=100
#NUM_IMGS=500
#NUM_IMGS=1000
#NUM_IMGS=5000
#NUM_IMGS=10000
#NUM_IMGS=50000
#NUM_IMGS=100000

for num_imgs in 100 500 1000 5000; do
  python data_generation/make_single_img_dataset.py \
    --img_size 256 \
    --batch_size 25 \
    --num_imgs ${num_imgs} \
    --threads 4 \
    --imgpath data_generation/images/ameyoko.jpg \
    --targetpath /data/fszatkowski/sid/ameyoko/${num_imgs}/
done;
