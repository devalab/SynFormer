#!/bin/bash

python -m molbart.fine_tune \
  --dataset uspto_50 \
  --data_path ../data/uspto50/uspto_50.pickle \
  --model_path None \
  --task backward_prediction \
  --epochs 1000 \
  --lr 0.001 \
  --schedule cycle \
  --batch_size 128 \
  --acc_batches 4 \
  --augment all \
  --aug_prob 0.5 \
  --gpus 1 \
  --d_model 512 \
  --num_layers 6 \
  --num_heads 8 \
  --d_feedforward 2048
