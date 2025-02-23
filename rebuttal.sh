#!/bin/bash
#SBATCH -A plafnet2
#SBATCH -p plafnet2
#SBATCH -J SynFormer
#SBATCH --output=/tmp/SynFormer.out
#SBATCH -c 20
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=3500
#SBATCH --time=4-00:00:00

mkdir -p /scratch/arihanth.srikar/
export PYTHONUNBUFFERED=1

python llm.py \
    --task SynFormer \
    --project uspto_mixed \
    --run test2 \
    --rotary_emb \
    --n_layer 6 \
    --n_head 8 \
    --n_embd 512 \
    --block_size 512 \
    --batch_size 64 \
    --grad_accum 8 \
    --vocab_size 576 \
    --learning_rate 0.001 \
    --lr_scheduler onecycle \
    --dividing_factor 10000 \
    --dropout 0.1 \
    --weight_decay 0.0 \
    --num_epochs 150 \
    --generate_every 50 \
    --ablation_res \
    --save_dir /scratch/arihanth.srikar \
    --set_precision \
    --train \
    --log
python llm.py \
    --task SynFormer \
    --project uspto_mixed \
    --run test2 \
    --rotary_emb \
    --n_layer 6 \
    --n_head 8 \
    --n_embd 512 \
    --block_size 512 \
    --batch_size 64 \
    --grad_accum 8 \
    --vocab_size 576 \
    --learning_rate 0.001 \
    --lr_scheduler onecycle \
    --dividing_factor 10000 \
    --dropout 0.1 \
    --weight_decay 0.0 \
    --num_epochs 150 \
    --generate_every 50 \
    --ablation_res \
    --set_precision \
    --save_dir /scratch/arihanth.srikar

# # best config retrain
# python llm.py \
#     --task SynFormer \
#     --project uspto50 \
#     --run best_config_retrain_2 \
#     --rotary_emb \
#     --n_layer 6 \
#     --n_head 8 \
#     --n_embd 512 \
#     --block_size 512 \
#     --batch_size 48 \
#     --grad_accum 2 \
#     --vocab_size 576 \
#     --learning_rate 0.001 \
#     --lr_scheduler onecycle \
#     --dividing_factor 10000 \
#     --dropout 0.1 \
#     --weight_decay 0.1 \
#     --num_epochs 1000 \
#     --generate_every 10 \
#     --ablation_res \
#     --save_dir /scratch/arihanth.srikar \
#     --train \
#     --log
# python llm.py \
#     --task SynFormer \
#     --project uspto50 \
#     --run best_config_retrain_2 \
#     --rotary_emb \
#     --n_layer 6 \
#     --n_head 8 \
#     --n_embd 512 \
#     --block_size 512 \
#     --batch_size 48 \
#     --grad_accum 2 \
#     --vocab_size 576 \
#     --learning_rate 0.001 \
#     --lr_scheduler onecycle \
#     --dividing_factor 10000 \
#     --dropout 0.1 \
#     --weight_decay 0.1 \
#     --num_epochs 1000 \
#     --generate_every 10 \
#     --ablation_res \
#     --save_dir /scratch/arihanth.srikar

# python llm.py \
#     --task SynFormer \
#     --project uspto50 \
#     --run best_config_retrain \
#     --rotary_emb \
#     --n_layer 6 \
#     --n_head 8 \
#     --n_embd 512 \
#     --block_size 512 \
#     --batch_size 48 \
#     --grad_accum 2 \
#     --vocab_size 576 \
#     --learning_rate 0.001 \
#     --lr_scheduler onecycle \
#     --dividing_factor 10000 \
#     --dropout 0.1 \
#     --weight_decay 0.1 \
#     --num_epochs 1000 \
#     --generate_every 10 \
#     --ablation_res \
#     --save_dir /scratch/arihanth.srikar

# # Ablation without SMILES augmentation
# python llm.py \
#     --task SynFormer \
#     --project uspto50 \
#     --run best_no_augment \
#     --rotary_emb \
#     --n_layer 6 \
#     --n_head 8 \
#     --n_embd 512 \
#     --block_size 512 \
#     --batch_size 48 \
#     --grad_accum 2 \
#     --vocab_size 576 \
#     --learning_rate 0.001 \
#     --lr_scheduler onecycle \
#     --dividing_factor 10000 \
#     --dropout 0.1 \
#     --weight_decay 0.1 \
#     --num_epochs 1000 \
#     --generate_every 10 \
#     --ablation_res \
#     --augment_fraction 0.0 \
#     --save_dir /scratch/arihanth.srikar \
#     --train \
#     --log
# python llm.py \
#     --task SynFormer \
#     --project uspto50 \
#     --run best_no_augment \
#     --rotary_emb \
#     --n_layer 6 \
#     --n_head 8 \
#     --n_embd 512 \
#     --block_size 512 \
#     --batch_size 48 \
#     --grad_accum 2 \
#     --vocab_size 576 \
#     --learning_rate 0.001 \
#     --lr_scheduler onecycle \
#     --dividing_factor 10000 \
#     --dropout 0.1 \
#     --weight_decay 0.1 \
#     --num_epochs 1000 \
#     --generate_every 10 \
#     --ablation_res \
#     --augment_fraction 0.0 \
#     --save_dir /scratch/arihanth.srikar
