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

python llm.py \
    --task pathformer \
    --project uspto50 \
    --run p1 \
    --rotary_emb \
    --n_layer 6 \
    --n_head 8 \
    --n_embd 512 \
    --block_size 512 \
    --batch_size 56 \
    --grad_accum 8 \
    --vocab_size 576 \
    --learning_rate 0.001 \
    --lr_scheduler onecycle \
    --dividing_factor 10000 \
    --dropout 0.1 \
    --weight_decay 0.1 \
    --num_epochs 1000 \
    --generate_every 10 \
    --save_dir /scratch/arihanth.srikar \
    --train \
    --log

python llm.py \
    --task pathformer \
    --project uspto50 \
    --run p2 \
    --rotary_emb \
    --n_layer 6 \
    --n_head 8 \
    --n_embd 512 \
    --block_size 512 \
    --batch_size 48 \
    --grad_accum 4 \
    --vocab_size 576 \
    --learning_rate 0.001 \
    --lr_scheduler onecycle \
    --dividing_factor 10000 \
    --dropout 0.1 \
    --weight_decay 0.1 \
    --num_epochs 1000 \
    --generate_every 10 \
    --save_dir /scratch/arihanth.srikar \
    --train \
    --log

python llm.py \
    --task pathformer \
    --project uspto50 \
    --run p3 \
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
    --num_epochs 1000 \
    --generate_every 10 \
    --save_dir /scratch/arihanth.srikar \
    --train \
    --log

python llm.py \
    --task pathformer \
    --project uspto50 \
    --run p4 \
    --rotary_emb \
    --n_layer 6 \
    --n_head 8 \
    --n_embd 512 \
    --block_size 512 \
    --batch_size 48 \
    --grad_accum 1 \
    --vocab_size 576 \
    --learning_rate 0.001 \
    --lr_scheduler onecycle \
    --dividing_factor 10000 \
    --dropout 0.1 \
    --weight_decay 0.1 \
    --num_epochs 1000 \
    --generate_every 10 \
    --save_dir /scratch/arihanth.srikar \
    --train \
    --log

python llm.py \
    --task pathformer \
    --project uspto50 \
    --run bpe \
    --rotary_emb \
    --n_layer 6 \
    --n_head 8 \
    --n_embd 512 \
    --block_size 512 \
    --batch_size 32 \
    --grad_accum 3 \
    --vocab_size 576 \
    --learning_rate 0.001 \
    --lr_scheduler onecycle \
    --dividing_factor 10000 \
    --dropout 0.1 \
    --weight_decay 0.1 \
    --num_epochs 1000 \
    --generate_every 10 \
    --ablation_res \
    --bpe_tokeniser \
    --save_dir /scratch/arihanth.srikar \
    --train \
    --log
python llm.py \
    --task pathformer \
    --project uspto50 \
    --run bpe \
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
    --num_epochs 1000 \
    --generate_every 10 \
    --ablation_res \
    --bpe_tokeniser \
    --save_dir /scratch/arihanth.srikar

python llm.py \
    --task pathformer \
    --project uspto50 \
    --run filtered_10 \
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
    --num_epochs 1000 \
    --generate_every 10 \
    --ablation_res \
    --filter_value 10 \
    --save_dir /scratch/arihanth.srikar \
    --train \
    --log
python llm.py \
    --task pathformer \
    --project uspto50 \
    --run filtered_10 \
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
    --num_epochs 1000 \
    --generate_every 10 \
    --ablation_res \
    --filter_value 10 \
    --save_dir /scratch/arihanth.srikar

python llm.py \
    --task pathformer \
    --project uspto50 \
    --run best_10 \
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
    --num_epochs 10 \
    --generate_every 1 \
    --ablation_res \
    --save_dir /scratch/arihanth.srikar \
    --train \
    --log
python llm.py \
    --task pathformer \
    --project uspto50 \
    --run best_10 \
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
    --num_epochs 10 \
    --generate_every 10 \
    --ablation_res \
    --save_dir /scratch/arihanth.srikar

python llm.py \
    --task pathformer \
    --project uspto50 \
    --run best_100 \
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
    --generate_every 10 \
    --ablation_res \
    --save_dir /scratch/arihanth.srikar \
    --train \
    --log
python llm.py \
    --task pathformer \
    --project uspto50 \
    --run best_100 \
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
    --generate_every 10 \
    --ablation_res \
    --save_dir /scratch/arihanth.srikar

python llm.py \
    --task pathformer \
    --project uspto50 \
    --run best_500 \
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
    --num_epochs 500 \
    --generate_every 10 \
    --ablation_res \
    --save_dir /scratch/arihanth.srikar \
    --train \
    --log
python llm.py \
    --task pathformer \
    --project uspto50 \
    --run best_500 \
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
    --num_epochs 500 \
    --generate_every 10 \
    --ablation_res \
    --save_dir /scratch/arihanth.srikar

python llm.py \
    --task pathformer \
    --project uspto50 \
    --run best_5000 \
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
    --num_epochs 5000 \
    --generate_every 10 \
    --ablation_res \
    --save_dir /scratch/arihanth.srikar \
    --train \
    --log
python llm.py \
    --task pathformer \
    --project uspto50 \
    --run best_5000 \
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
    --num_epochs 5000 \
    --generate_every 10 \
    --ablation_res \
    --save_dir /scratch/arihanth.srikar

python llm.py \
    --task pathformer \
    --project uspto50 \
    --run p3_with_residual_no_cross_attn_res \
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
    --num_epochs 1000 \
    --generate_every 10 \
    --residual \
    --ablation_res \
    --save_dir /scratch/arihanth.srikar \
    --train \
    --log
python llm.py \
    --task pathformer \
    --project uspto50 \
    --run p3_with_residual_no_cross_attn_res \
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
    --num_epochs 1000 \
    --generate_every 10 \
    --residual \
    --ablation_res \
    --device_ids 0 \
    --save_dir /scratch/arihanth.srikar

python llm.py \
    --task pathformer \
    --project uspto50 \
    --run p3_ablation_pos_emb \
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
    --num_epochs 1000 \
    --generate_every 10 \
    --ablation_pos_emb \
    --save_dir /scratch/arihanth.srikar \
    --train \
    --log
python llm.py \
    --task pathformer \
    --project uspto50 \
    --run p3_ablation_pos_emb \
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
    --num_epochs 1000 \
    --generate_every 10 \
    --ablation_pos_emb \
    --device_ids 0 \
    --save_dir /scratch/arihanth.srikar


python llm.py \
    --task pathformer \
    --project uspto50 \
    --run p3_ablation_act_fn \
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
    --num_epochs 1000 \
    --generate_every 10 \
    --ablation_act_fn \
    --save_dir /scratch/arihanth.srikar \
    --train \
    --log
python llm.py \
    --task pathformer \
    --project uspto50 \
    --run p3_ablation_act_fn \
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
    --num_epochs 1000 \
    --generate_every 10 \
    --ablation_act_fn \
    --device_ids 0 \
    --save_dir /scratch/arihanth.srikar


python llm.py \
    --task pathformer \
    --project uspto50 \
    --run p3_ablation_res \
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
    --num_epochs 1000 \
    --generate_every 10 \
    --ablation_res \
    --save_dir /scratch/arihanth.srikar \
    --train \
    --log
python llm.py \
    --task pathformer \
    --project uspto50 \
    --run p3_ablation_res \
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
    --num_epochs 1000 \
    --generate_every 10 \
    --ablation_res \
    --device_ids 0 \
    --save_dir /scratch/arihanth.srikar


python llm.py \
    --task pathformer \
    --project uspto50 \
    --run p3_with_residual \
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
    --num_epochs 1000 \
    --generate_every 10 \
    --residual \
    --save_dir /scratch/arihanth.srikar \
    --train \
    --log
python llm.py \
    --task pathformer \
    --project uspto50 \
    --run p3_with_residual \
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
    --num_epochs 1000 \
    --generate_every 10 \
    --residual \
    --device_ids 0 \
    --save_dir /scratch/arihanth.srikar
