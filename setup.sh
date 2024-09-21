conda create -y -n synformer python=3.10
conda activate synformer
conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

# pip dependencies
pip install pytorch-lightning==2.0.3 tqdm rdkit-pypi pandas einops prettytable transformer wandb numpy==1.24.3
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
python -m pip install git+https://github.com/MolecularAI/pysmilesutils.git