# SynFormer
An encoder-decoder transformer model that models retrosynthetic analysis along with novel metrics for evaluation. We use the [X-transformer](https://github.com/lucidrains/x-transformers/tree/main) library and the dataloader from [Chemformer](https://github.com/MolecularAI/Chemformer/tree/main).

`llm.py` contains the code to train and evaluate the model. All commands, including the ablation studies, are in `llm.sh`. The dataset can be found at `data/uspto50/uspto_50.pickle`.

The code to our metric - Retrosynthesis Refinement Index (RRI), can be found at `benchmark.py,`, and the results of analyzing various algorithms with our metric can be found at `benchmark.ipynb`.

Create the environment `conda create -y -n synformer python=3.10`, activate the environment `conda activate synformer` and install the dependencies:
 - `conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia`
 - `pip install pytorch-lightning==2.0.3 rdkit-pypi pandas einops prettytable transformer wandb numpy==1.24.3`
 - `pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git`
 - `python -m pip install git+https://github.com/MolecularAI/pysmilesutils.git`

Our model can be trained using the following command:
```python
python llm.py \
    --task SynFormer \
    --project uspto50 \
    --run SynFormer \
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
```

Our model can be evaluated using the following command:
```python
python llm.py \
    --task SynFormer \
    --project uspto50 \
    --run SynFormer \
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
    --save_dir /scratch/arihanth.srikar
```
