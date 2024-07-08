#!/bin/bash
#SBATCH -A plafnet2
#SBATCH -p plafnet2
#SBATCH -J SynFormer
#SBATCH --output=/tmp/SynFormer.out
#SBATCH -c 20
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=3000
#SBATCH --time=4-00:00:00

mkdir -p /scratch/arihanth.srikar/uspto50
export PYTHONUNBUFFERED=1

cp -r ~/p3_ablation_res /scratch/arihanth.srikar/uspto50

# evaluate best config so far with beam search
python llm.py \
    --task SynFormer \
    --project uspto50 \
    --run p3_ablation_res \
    --rotary_emb \
    --n_layer 6 \
    --n_head 8 \
    --n_embd 512 \
    --block_size 512 \
    --batch_size 1 \
    --grad_accum 2 \
    --vocab_size 576 \
    --learning_rate 0.001 \
    --lr_scheduler onecycle \
    --dividing_factor 10000 \
    --dropout 0.1 \
    --weight_decay 0.1 \
    --num_epochs 1000 \
    --generate_every 10 \
    --ablation_res \
    --device_ids 0 \
    --beam_width 10 \
    --save_dir /scratch/arihanth.srikar