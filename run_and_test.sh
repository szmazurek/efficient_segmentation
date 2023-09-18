#!/bin/bash

# build the docker image

docker build -t efficient_segmentation .

# run the training
mkdir -p $6
mkdir -p $1/$5
docker run -v ./$1:/efficient_segmentation/data \
    -m=25G --shm-size=20G \
    -v ./$6:/efficient_segmentation/checkpoints \
    efficient_segmentation \
    --train \
    --training_data_path $1/$2/ \
    --validation_data_path $1/$3/ \
    --testing_data_path  $1/$4/ \
    --test_results_save_path $1/$5/ \
    --model_path $6/best_model.ckpt \
    --lr 0.001 \
    --num_classes 1 \
    --epochs 1 \
    --img_size 256 \
    --batch_size 128 \
    --model AttSqueezeUnet \
    --loss_function DiceLoss

docker run -it -v ./$1:/efficient_segmentation/data \
    -v ./$6:/efficient_segmentation/checkpoints/ \
    -m=25G --shm-size=20G  efficient_segmentation \
    --test \
    --training_data_path $1/$2/ \
    --validation_data_path $1/$3/ \
    --testing_data_path  $1/$4/ \
    --test_results_save_path $1/$5/ \
    --model_path $6/best_model.ckpt \
    --lr 0.001 \
    --num_classes 1 \
    --epochs 150   \
    --img_size 256 \
    --batch_size 1 \
    --model AttSqueezeUnet \
    --loss_function DiceLoss
