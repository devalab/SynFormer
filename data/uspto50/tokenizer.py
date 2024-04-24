import pandas as pd
from SmilesPE.pretokenizer import atomwise_tokenizer


df = pd.read_pickle('processed.pickle')
print(df.head())

vocab = dict()
for entry in df['reactants_mol']:
    smi = '.'.join(entry)
    toks = atomwise_tokenizer(smi)
    for tok in toks:
        if tok in vocab:
            vocab[tok] += 1
        else:
            vocab[tok] = 1
print(f'vocab size: {len(vocab)}')

for entry in df['products_mol']:
    smi = '.'.join(entry)
    toks = atomwise_tokenizer(smi)
    for tok in toks:
        if tok in vocab:
            vocab[tok] += 1
        else:
            vocab[tok] = 1
print(f'vocab size: {len(vocab)}')

with open('vocab.txt', 'w') as f:
    for k, v in vocab.items():
        f.write(f'{k}\n')
    f.write('<unk>\n')
    f.write('<sor>\n')
    f.write('<eor>\n')
    f.write('<sop>\n')
    f.write('<eop>\n')
    f.write('<mask>\n')
    f.write('<sum_pred>\n')
    f.write('<sum_react>\n')
    f.write('<first_pred>\n')
    f.write('<second_pred>\n')
    f.write('<third_pred>\n')
    f.write('<fourth_pred>\n')
    f.write('<pad>\n')
