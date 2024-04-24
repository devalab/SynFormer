#!/bin/bash

python -m molbart.predict \
  --input_path data/chemformer_input_test.txt \
  --target_path data/chemformer_target_test.txt \
  --output_path data/rand_init_finetuned_uspto50.pickle \
  --model_path data/last.ckpt \
  --batch_size 64 \
  --num_beams 10

