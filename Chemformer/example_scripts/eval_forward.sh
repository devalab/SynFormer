#!/bin/bash

# forward prediction
python -m molbart.evaluate \
  --data_path ../data/uspto_sep.pickle \
  --model_path saved_models/uspto_sep/span_aug/100_epochs/last.ckpt \
  --dataset uspto_sep \
  --task forward_prediction \
  --model_type bart \
  --batch_size 64 \
  --num_beams 10


# backward prediction
python -m molbart.evaluate \
  --data_path ../data/uspto50/uspto_50.pickle \
  --model_path /scratch/arihanth.srikar/ \
  --dataset uspto50 \
  --task backward_prediction \
  --model_type bart \
  --batch_size 64 \
  --num_beams 10

