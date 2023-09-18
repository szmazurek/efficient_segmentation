#!/bin/bash

# build the docker image

docker build -t efficient_segmentation .

# run the training
mkdir -p $6

docker run -v ./$1:/efficient_segmentation/data \
    -m=25G --shm-size=20G \
    -v ./$6:/efficient_segmentation/checkpoints \
    efficient_segmentation \
    --train \
    --training_data_path data/$2/ \
    --validation_data_path data/$3/ \
    --testing_data_path  data/$4/ \
    --test_results_save_path data/$5/ \
    --model_path $6/best_model.ckpt \
    --lr 0.001 \
    --num_classes 1 \
    --epochs 1 \
    --img_size 256 \
    --batch_size 16 \
    --model AttSqueezeUnet \
    --loss_function DiceLoss

# run the testing

docker run -v ./$S1:/efficient_segmentation/data \
    -v ./$6:/efficient_segmentation/checkpoints \
    -m=25G --shm-size=20G \
    efficient_segmentation \
    --test \
    --training_data_path data/$2/ \
    --validation_data_path data/$3/ \
    --testing_data_path  data/$4/ \
    --test_results_save_path data/$5/ \
    --model_path $6/best_model.ckpt \
    --lr 0.001 \
    --num_classes 1 \
    --epochs 150 \
    --img_size 256 \
    --batch_size 1 \
    --model AttSqueezeUnet \
    --loss_function DiceLoss