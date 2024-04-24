#!/bin/bash
#SBATCH -J llm
#SBATCH --output=logs/data.out
#SBATCH -c 40
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2900
#SBATCH --constraint=2080ti
#SBATCH --time=4-00:00:00

mkdir -p /scratch/arihanth.srikar/
export PYTHONUNBUFFERED=1

scp ada:/share1/arihanth.srikar/uspto_mixed.pickle /scratch/arihanth.srikar
scp ada:/share1/arihanth.srikar/zinc.zip /scratch/arihanth.srikar
unzip /scratch/arihanth.srikar/zinc.zip -d /scratch/arihanth.srikar

python llm.py \
    --task pathformer \
    --project uspto_mixed \
    --run test1 \
    --rotary_emb \
    --n_layer 6 \
    --n_head 8 \
    --n_embd 512 \
    --block_size 512 \
    --batch_size 48 \
    --grad_accum 2 \
    --vocab_size 576 \
    --learning_rate 0.001 \
    --lr_scheduler onecycle \
    --dividing_factor 10000 \
    --dropout 0.1 \
    --weight_decay 0.1 \
    --num_epochs 100 \
    --generate_every 1 \
    --ablation_res \
    --save_dir /scratch/arihanth.srikar \
    --train \
    --log
python llm.py \
    --task pathformer \
    --project uspto_mixed \
    --run test1 \
    --rotary_emb \
    --n_layer 6 \
    --n_head 8 \
    --n_embd 512 \
    --block_size 512 \
    --batch_size 48 \
    --grad_accum 2 \
    --vocab_size 576 \
    --learning_rate 0.001 \
    --lr_scheduler onecycle \
    --dividing_factor 10000 \
    --dropout 0.1 \
    --weight_decay 0.1 \
    --num_epochs 100 \
    --generate_every 1 \
    --ablation_res \
    --save_dir /scratch/arihanth.srikar

# python llm.py \
#     --task pathformer \
#     --project uspto50 \
#     --run p1 \
#     --rotary_emb \
#     --n_layer 6 \
#     --n_head 8 \
#     --n_embd 512 \
#     --block_size 512 \
#     --batch_size 56 \
#     --grad_accum 8 \
#     --vocab_size 576 \
#     --learning_rate 0.001 \
#     --lr_scheduler onecycle \
#     --dividing_factor 10000 \
#     --dropout 0.1 \
#     --weight_decay 0.1 \
#     --num_epochs 1000 \
#     --generate_every 10 \
#     --save_dir /scratch/arihanth.srikar \
#     --train \
#     --log

# python llm.py \
#     --task pathformer \
#     --project uspto50 \
#     --run p2 \
#     --rotary_emb \
#     --n_layer 6 \
#     --n_head 8 \
#     --n_embd 512 \
#     --block_size 512 \
#     --batch_size 48 \
#     --grad_accum 4 \
#     --vocab_size 576 \
#     --learning_rate 0.001 \
#     --lr_scheduler onecycle \
#     --dividing_factor 10000 \
#     --dropout 0.1 \
#     --weight_decay 0.1 \
#     --num_epochs 1000 \
#     --generate_every 10 \
#     --save_dir /scratch/arihanth.srikar \
#     --train \
#     --log

# python llm.py \
#     --task pathformer \
#     --project uspto50 \
#     --run p3 \
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
#     --save_dir /scratch/arihanth.srikar \
#     --train \
#     --log

# python llm.py \
#     --task pathformer \
#     --project uspto50 \
#     --run p4 \
#     --rotary_emb \
#     --n_layer 6 \
#     --n_head 8 \
#     --n_embd 512 \
#     --block_size 512 \
#     --batch_size 48 \
#     --grad_accum 1 \
#     --vocab_size 576 \
#     --learning_rate 0.001 \
#     --lr_scheduler onecycle \
#     --dividing_factor 10000 \
#     --dropout 0.1 \
#     --weight_decay 0.1 \
#     --num_epochs 1000 \
#     --generate_every 10 \
#     --save_dir /scratch/arihanth.srikar \
#     --train \
#     --log
