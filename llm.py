import os
import gc
import math
import argparse
from glob import glob
from tqdm import tqdm
from rdkit import Chem
import pandas as pd

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import AdamW

from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from warmup_scheduler import GradualWarmupScheduler

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from x_transformers.x_transformers import XTransformer
from x_transformers.autoregressive_wrapper import top_k

import Chemformer.molbart.util as util
from Chemformer.molbart.data.datasets import Uspto50, UsptoMixed
from Chemformer.molbart.data.datamodules import FineTuneReactionDataModule, RetroDataModule
from benchmark import Metrics

from transformers import AutoTokenizer

# disable rdkit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

parser = argparse.ArgumentParser(description='Retrosynthesis')

parser.add_argument('--block_size', type=int, default=512, help='block size')
parser.add_argument('--vocab_size', type=int, default=530, help='vocab size')
parser.add_argument('--n_layer', type=int, default=6, help='number of layers')
parser.add_argument('--n_head', type=int, default=8, help='number of heads')
parser.add_argument('--n_embd', type=int, default=512, help='embedding dimension')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
parser.add_argument('--bias', action=argparse.BooleanOptionalAction, help='whether to use bias in attention layer')
parser.add_argument('--rotary_emb', action=argparse.BooleanOptionalAction, help='whether to use rotary embeddings')
parser.add_argument('--add_attn_z_loss', action=argparse.BooleanOptionalAction, help='wether to add attn z_loss')
parser.add_argument('--beam_width', type=int, default=1, help='beam width')

parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--num_epochs', type=int, default=1000, help='number of epochs')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2')
parser.add_argument('--weight_decay', type=float, default=1e-1, help='weight decay')
parser.add_argument('--lr_scheduler', type=str, default='onecycle', help='lr scheduler')
parser.add_argument('--dividing_factor', type=float, default=10000, help='dividing factor for lr scheduler')
parser.add_argument('--augment_fraction', type=float, default=0.5, help='filter value for dataset')
parser.add_argument('--filter_value', type=int, default=0, help='filter value for dataset')
parser.add_argument('--bpe_tokeniser', action=argparse.BooleanOptionalAction, help='whether to use bpe tokeniser')

parser.add_argument('--data_dir', type=str, default='data/', help='data directory')
parser.add_argument('--validate_every', type=int, default=500, help='train iterations')
parser.add_argument('--validate_for', type=int, default=100, help='validate iterations')
parser.add_argument('--generate_every', type=int, default=10, help='interval to generate')
parser.add_argument('--generate_for', type=int, default=2, help='generate iterations')
parser.add_argument('--train', action=argparse.BooleanOptionalAction, help='whether to train the model')
parser.add_argument('--grad_accum', type=int, default=4, help='gradient accumulation')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers')

parser.add_argument('--device', type=str, default='cuda', help='device')
parser.add_argument('--is_compile', action=argparse.BooleanOptionalAction, help='whether to compile the model')
parser.add_argument('--task', type=str, default='uspto50', help='task')
parser.add_argument('--run', type=str, default='exp', help='run name')
parser.add_argument('--project', type=str, default='uspto50', help='project name')
parser.add_argument('--entity', type=str, default='retrosynthesis', help='entity name')
parser.add_argument('--save_dir', type=str, default='/scratch/arihanth.srikar', help='save directory')
parser.add_argument('--log', action=argparse.BooleanOptionalAction, help='whether to log')
parser.add_argument('--set_precision', action=argparse.BooleanOptionalAction, help='whether to set precision')
parser.add_argument('--device_ids', type=int, nargs='*', help='device ids')
parser.add_argument('--vocab_file', type=str, default='', help='vocab files')
parser.add_argument('--sub_task', type=str, default='dec', help='sub task')
parser.add_argument('--load_from', type=str, default='', help='load checkpoint from')

parser.add_argument('--ablation_pos_emb', action=argparse.BooleanOptionalAction, help='ablation study on positional embeddings')
parser.add_argument('--ablation_act_fn', action=argparse.BooleanOptionalAction, help='ablation study on activation function')
parser.add_argument('--ablation_res', action=argparse.BooleanOptionalAction, help='ablation study on cross attention residual')
parser.add_argument('--residual', action=argparse.BooleanOptionalAction, help='add residuals')

config = vars(parser.parse_args())
config["data_dir"] = config["data_dir"] + config["task"]


class SynFormer(pl.LightningModule):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config

        self.token_encode = {v: k for k, v in enumerate(config['vocab'])} if type(config['vocab']) == list else config['vocab']
        self.token_decode = {k: v for k, v in enumerate(config['vocab'])} if type(config['vocab']) == list else {v: k for k, v in config['vocab'].items()}

        self.llm = XTransformer(
            dim = config['n_embd'],
            
            # encoder args
            enc_num_tokens = config['vocab_size'],
            enc_depth = config['n_layer'],
            enc_heads = config['n_head'],
            enc_max_seq_len = config['block_size'],
            enc_layer_dropout = config['dropout'],
            enc_attn_dropout = config['dropout'],
            enc_ff_dropout = config['dropout'],
            enc_ff_relu_squared = not config['ablation_act_fn'],
            enc_ff_no_bias = True,
            enc_rotary_pos_emb = not config['ablation_pos_emb'],
            enc_residual_attn = config['residual'],

            # decoder args
            dec_num_tokens = config['vocab_size'],
            dec_depth = config['n_layer'],
            dec_heads = config['n_head'],
            dec_max_seq_len = config['block_size'],
            dec_layer_dropout = config['dropout'],
            dec_attn_dropout = config['dropout'],
            dec_ff_dropout = config['dropout'],
            dec_ff_relu_squared = not config['ablation_act_fn'],
            dec_ff_no_bias = True,
            dec_cross_residual_attn = not config['ablation_res'],
            dec_residual_attn = config['residual'],
            dec_rotary_pos_emb = not config['ablation_pos_emb'],
            
            # general args
            ignore_index=config['pad_token_id'],
            pad_value=config['pad_token_id'],
            bos_value=config['begin_token_id'],
            eos_value=config['end_token_id'],
            cross_attn_tokens_dropout=0.0,
            tie_token_emb = True,
        )

        self.save_hyperparameters()

        self.val_epoch_end_outputs = []
        self.test_epoch_end_outputs = []

    def forward(self, batch):
        products  = batch["encoder_input"].transpose(0, 1)
        reactants = batch["decoder_input"].transpose(0, 1)
        src_mask  = products != self.config['pad_token_id']

        return self.llm(src=products, tgt=reactants, mask=src_mask)

    def training_step(self, batch, batch_idx):
        loss = self(batch)

        # get learning rate
        lr = self.optimizers().param_groups[0]['lr']
        
        self.log('lr', lr, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        _, target, predicted = self.generate(batch)
        self.val_epoch_end_outputs.extend(list(zip(target, predicted)))

    def test_step(self, batch, batch_idx):
        _, target, predicted = self.generate(batch)
        self.test_epoch_end_outputs.extend(list(zip(target, predicted)))

    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
        target_list, predicted_list = zip(*self.val_epoch_end_outputs)
        LLM_metrics = Metrics(target_list, predicted_list, f'Validating at {self.current_epoch} epoch')
        metrics = LLM_metrics.get_metrics()

        for k,v in metrics.items():
            self.log(f'val_{k}', v, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    
        self.val_epoch_end_outputs = []

    @torch.no_grad()
    def on_test_epoch_end(self) -> None:
        target_list, predicted_list = zip(*self.test_epoch_end_outputs)
        LLM_metrics = Metrics(target_list, predicted_list, f'Testing at {self.current_epoch} epoch')
        metrics = LLM_metrics.get_metrics()

        for k,v in metrics.items():
            self.log(f'test_{k}', v, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    
        self.test_epoch_end_outputs = []

    @torch.no_grad()
    def generate(self, batch):
        products  = batch["encoder_input"].transpose(0, 1)
        reactants = batch["decoder_input"].transpose(0, 1)
        src_mask  = products != self.config['pad_token_id']

        gen_seq = self.llm.generate(seq_in=products, seq_out_start=reactants[:, :1], seq_len=reactants.shape[1], 
                                    mask=src_mask, eos_token=self.config['end_token_id'], filter_kwargs={'k': 1})

        if not self.config['bpe_tokeniser']:
            prods = [''.join([self.token_decode[t.item()] for t in prod if t.item() != self.config['pad_token_id']])[1:-1].split('.') for prod in products]
            reacts = [''.join([self.token_decode[t.item()] for t in react if t.item() != self.config['pad_token_id']])[1:-1].split('.') for react in reactants]
            gens = [''.join([self.token_decode[t.item()] for t in gen if t.item() != self.config['pad_token_id']]) for gen in gen_seq]
            gens = [g[:g.find(config['end_token'])].split('.') if g.find(config['end_token']) != -1 else g.split('.') for g in gens]
        else:
            prods = [t.split('.') for t in tokeniser.batch_decode(products, skip_special_tokens=True)]
            reacts = [t.split('.') for t in  tokeniser.batch_decode(reactants, skip_special_tokens=True)]
            gens = [t.split('.') for t in tokeniser.batch_decode(gen_seq, skip_special_tokens=True)]
        
        return prods, reacts, gens
    
    @torch.no_grad()
    def beam_generate(self, batch):
        products  = batch["encoder_input"].transpose(0, 1)
        reactants = batch["decoder_input"].transpose(0, 1)
        src_mask  = products != self.config['pad_token_id']

        gen_seq = self.llm.beam_generate(seq_in=products, seq_out_start=reactants[:, :1], seq_len=reactants.shape[1], 
                                    mask=src_mask, eos_token=self.config['end_token_id'], filter_kwargs={'k': 1}, 
                                    num_beams=self.config['beam_width'])
        gen_seq = gen_seq.view(self.config['beam_width'], -1)

        prods = [''.join([self.token_decode[t.item()] for t in prod if t.item() != self.config['pad_token_id']])[1:-1].split('.') for prod in products]
        reacts = [''.join([self.token_decode[t.item()] for t in react if t.item() != self.config['pad_token_id']])[1:-1].split('.') for react in reactants]
        gens = [''.join([self.token_decode[t.item()] for t in gen if t.item() != self.config['pad_token_id']]) for gen in gen_seq]
        gens = [g[:g.find(config['end_token'])].split('.') if g.find(config['end_token']) != -1 else g.split('.') for g in gens]
        
        return prods, reacts, gens

    @torch.no_grad()
    def beam_generate_old(self, batch):
        products  = batch["encoder_input"].transpose(0, 1)
        reactants = batch["decoder_input"].transpose(0, 1)
        src_mask  = products != self.config['pad_token_id']

        beam_width = self.config['beam_width']
        gen_seq, probs = self.llm.beam_generate_old(seq_in=products, mask=src_mask, beam_width=beam_width)

        if not self.config['bpe_tokeniser']:
            gen_seq = gen_seq[:, 1:]
            prods = [''.join([self.token_decode[t.item()] for t in prod if t.item() != self.config['pad_token_id']])[1:-1].split('.') for prod in products]
            reacts = [''.join([self.token_decode[t.item()] for t in react if t.item() != self.config['pad_token_id']])[1:-1].split('.') for react in reactants]
            gens = [''.join([self.token_decode[t.item()] for t in gen if t.item() != self.config['pad_token_id']]) for gen in gen_seq]
            gens = [g[:g.find(config['end_token'])].split('.') if g.find(config['end_token']) != -1 else g.split('.') for g in gens]
        else:
            prods = tokeniser.batch_decode(products, skip_special_tokens=True).split('.')
            reacts = tokeniser.batch_decode(reactants, skip_special_tokens=True).split('.')
            gens = tokeniser.batch_decode(gen_seq, skip_special_tokens=True).split('.')
        
        return prods, reacts, gens
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.config['learning_rate'], 
            betas=(self.config['beta1'], self.config['beta2']), 
            weight_decay=self.config['weight_decay'], 
            )
        
        if self.config['lr_scheduler'] == 'cosine':
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.config['num_steps'], eta_min=self.config['learning_rate']/50)
        elif self.config['lr_scheduler'] == 'cosine_warmup':
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.config['num_steps'], eta_min=self.config['learning_rate']/50)
            lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=self.config['num_steps']//10, after_scheduler=lr_scheduler)
        elif self.config['lr_scheduler'] == 'onecycle':
            lr_scheduler = OneCycleLR(optimizer, max_lr=self.config['learning_rate'], total_steps=self.config['num_steps'])
        else:
            raise NotImplementedError
        
        scheduler = {"scheduler": lr_scheduler, "interval": "step"}
        
        return [optimizer], scheduler


if __name__ == '__main__':
    if config['set_precision']:
        torch.set_float32_matmul_precision('medium')
    torch.cuda.empty_cache()
    gc.collect()

    print("Building tokeniser...")
    tokeniser = util.load_tokeniser('data/uspto50/my_vocab.txt', 272) if not config['bpe_tokeniser'] else AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_396_250")
    print("Finished tokeniser.")

    with open('data/uspto50/my_vocab.txt', 'r') as f:
        vocab = f.read().split('\n')
    config['vocab'] = vocab if not config['bpe_tokeniser'] else tokeniser.get_vocab()

    config['pad_token_id'] = tokeniser.vocab[tokeniser.pad_token] if not config['bpe_tokeniser'] else tokeniser.pad_token_id
    config['mask_token_id'] = tokeniser.vocab[tokeniser.mask_token] if not config['bpe_tokeniser'] else tokeniser.mask_token_id
    config['begin_token_id'] = tokeniser.vocab[tokeniser.begin_token] if not config['bpe_tokeniser'] else tokeniser.bos_token_id
    config['end_token_id'] = tokeniser.vocab[tokeniser.end_token] if not config['bpe_tokeniser'] else tokeniser.eos_token_id
    config['sep_token_id'] = tokeniser.vocab[tokeniser.sep_token] if not config['bpe_tokeniser'] else tokeniser.sep_token_id
    config['end_token'] = '&' if not config['bpe_tokeniser'] else tokeniser.eos_token
    config['vocab_size'] = config['vocab_size'] if not config['bpe_tokeniser'] else tokeniser.vocab_size

    print(f'Vocab Size: {config["vocab_size"]}')

    print("Reading dataset...")
    dataset_filename = 'data/uspto50/uspto_50.pickle' if config['filter_value'] == 0 else f'data/uspto50/uspto_50_filtered_{config["filter_value"]}.pickle'
    if config['project'] == 'uspto50':
        dataset = Uspto50(dataset_filename, config['augment_fraction'], forward=False)
    elif config['project'] == 'uspto_mixed':
        dataset = UsptoMixed('data/uspto_mixed/uspto_mixed.pickle', config['augment_fraction'])
    print("Finished dataset.")

    print("Building data module...")
    if not config['bpe_tokeniser']:
        dm = FineTuneReactionDataModule(
                dataset,
                tokeniser,
                config['batch_size'],
                config['block_size'],
                forward_pred=False,
                val_idxs=dataset.val_idxs,
                test_idxs=dataset.test_idxs,
                train_token_batch_size=None,
                num_buckets=24,
                unified_model=False,
            )
    else:
        dm = RetroDataModule(
                dataset,
                tokeniser,
                config['batch_size'],
                config['block_size'],
                forward_pred=False,
                val_idxs=dataset.val_idxs,
                test_idxs=dataset.test_idxs,
                train_token_batch_size=None,
                num_buckets=24,
                unified_model=False,
            )
    num_available_cpus = len(os.sched_getaffinity(0))
    num_available_gpus = torch.cuda.device_count()
    num_workers = num_available_cpus // num_available_gpus
    dm._num_workers = num_workers
    print(f"Using {str(num_workers)} workers for data module.")
    print("Finished datamodule.")

    dm.setup()
    batches_per_gpu = math.ceil(len(dm.train_dataloader()) / num_available_gpus)
    train_steps = math.ceil(batches_per_gpu / config['grad_accum']) * config['num_epochs']
    config['num_steps'] = train_steps

    model = SynFormer(config)

    logger = WandbLogger(
        # entity=config['entity'],
        project=config['project'],
        name=config['run'],
        save_dir=config['save_dir'],
        mode='disabled' if not config['log'] else 'online',
    )

    accuracy_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        monitor="val_accuracy",
        mode="max",
        dirpath=f"{config['save_dir']}/{config['project']}/{config['run']}",
        filename="model-{epoch:02d}-{val_accuracy:.5f}",
    )
    accuracy_callback.CHECKPOINT_NAME_LAST = "{epoch:02d}-last"

    adjusted_accuracy_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_adjusted_accuracy",
        mode="max",
        dirpath=f"{config['save_dir']}/{config['project']}/{config['run']}",
        filename="model-{epoch:02d}-{val_adjusted_accuracy:.5f}",
    )

    partial_accuracy_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_partial_accuracy",
        mode="max",
        dirpath=f"{config['save_dir']}/{config['project']}/{config['run']}",
        filename="model-{epoch:02d}-{val_partial_accuracy:.5f}",
    )

    score_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_score",
        mode="max",
        dirpath=f"{config['save_dir']}/{config['project']}/{config['run']}",
        filename="model-{epoch:02d}-{val_score:.5f}",
    )

    score_hc_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_score_hc",
        mode="max",
        dirpath=f"{config['save_dir']}/{config['project']}/{config['run']}",
        filename="model-{epoch:02d}-{val_score_hc:.5f}",
    )

    index_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_index",
        mode="max",
        dirpath=f"{config['save_dir']}/{config['project']}/{config['run']}",
        filename="model-{epoch:02d}-{val_index:.5f}",
    )
    
    model_callbacks = [accuracy_callback, adjusted_accuracy_callback, partial_accuracy_callback, 
                       score_callback, score_hc_callback, index_callback]

    trainer = pl.Trainer(
        accelerator='gpu', devices=-1, strategy='ddp_find_unused_parameters_True',
        # accelerator='gpu', devices=-1, strategy='auto',
        max_epochs=config['num_epochs'], logger=logger,
        precision='bf16-mixed' if config['set_precision'] else '32-true',
        gradient_clip_val=0.5, gradient_clip_algorithm='norm',
        accumulate_grad_batches=config['grad_accum'],
        callbacks=model_callbacks,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        check_val_every_n_epoch=config['generate_every'],
    )

    if config['train']:
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm)

    else:
        if config['beam_width'] > 1:
            assert config['batch_size'] == 1, "Batch size must be 1 for beam search"

        # manually load data
        dm.setup()

        device = f'cuda'
        model_criteria = ['last', 'val_accuracy', 'val_partial_accuracy', 'val_index']
        model_criteria = ['last']
        model_ckpt = sorted(glob(f"{config['save_dir']}/{config['project']}/{config['run']}/*.ckpt"))

        for criteria in model_criteria:
            try:
                ckpt = [ckpt for ckpt in model_ckpt if criteria in ckpt][-1]
            except:
                continue
            model = SynFormer.load_from_checkpoint(ckpt, config=config)
            print(f"Loaded model from {ckpt}")
            model = model.to(device)
            model = model.eval()

            # generate sequences
            # for (split, split_dm) in [('val', dm.val_dataloader()), ('test', dm.test_dataloader())]:
            for (split, split_dm) in [('test', dm.test_dataloader())]:
                # dump the predicted and actual products to a pandas dataframe
                df = pd.DataFrame(columns=['target_smiles', 'predicted_smiles', 'input_smiles'])
                input_smiles = []
                target_smiles = []
                predicted_smiles = []

                with torch.no_grad():
                    for batch_id, batch in enumerate(tqdm(split_dm, desc=f'Generating {split} sequences')):
                        batch = {k: v.to(device) for k, v in batch.items() if 'input' in k}
                        
                        prods, reacts, gens = model.generate(batch) if config['beam_width'] == 1 else model.beam_generate(batch)

                        input_smiles.extend(prods)
                        target_smiles.extend(reacts)
                        predicted_smiles.extend(gens) if config['beam_width'] == 1 else predicted_smiles.extend([gens])
                        
                # update dataframe
                df['input_smiles'] = input_smiles
                df['target_smiles'] = target_smiles
                if config['beam_width'] == 1:
                    df['predicted_smiles'] = predicted_smiles
                else:
                    for beam_id in range(config['beam_width']):
                        print(f'predicted_smiles_{beam_id}: {[pred[beam_id] for pred in predicted_smiles]}')
                        df[f'predicted_smiles_{beam_id}'] = [pred[beam_id] for pred in predicted_smiles]
                print(df)

                try:
                    # get metrics
                    LLM_metrics = Metrics(target_smiles, predicted_smiles if config['beam_width'] == 1 else [pred[0] for pred in predicted_smiles], f'{config["run"]}_{split}_{criteria}_beam_{config["beam_width"]}')
                    metrics = LLM_metrics.get_metrics()
                    print(LLM_metrics.print_metrics())
                except:
                    pass
                
                # save dataframe and metrics
                # save_dir = f'{config["save_dir"]}/results'
                save_dir = f'results'
                os.makedirs(f'{save_dir}', exist_ok=True)
                os.makedirs(f'{save_dir}/{config["run"]}', exist_ok=True)
                
                df.to_csv(f"{save_dir}/{config['run']}/{split}_{criteria}_beam_{config['beam_width']}.csv", index=False)
                with open(f"{save_dir}/{config['run']}/{split}_{criteria}_beam_{config['beam_width']}.txt", 'w') as f:
                    print(LLM_metrics.print_metrics(), file=f)
            