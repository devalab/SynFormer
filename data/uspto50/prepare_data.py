import pandas as pd
from tqdm import tqdm
from SmilesPE.pretokenizer import atomwise_tokenizer


with open('vocab.txt', 'r') as f:
    idx2char = f.read().splitlines()
char2idx = {k: v for v, k in enumerate(idx2char)}


df = pd.read_pickle('processed.pickle')
reactants = df['reactants_mol'].apply(lambda x: '.'.join(x))
products  = df['products_mol'].apply(lambda x: '.'.join(x))
r_tokens = []
p_tokens = []

for r, p in tqdm(zip(reactants, products), total=len(reactants)):
    r_toks = atomwise_tokenizer(r)
    p_toks = atomwise_tokenizer(p)
    r_idxs = [char2idx[tok] for tok in r_toks]
    p_idxs = [char2idx[tok] for tok in p_toks]
    # r_idxs = [char2idx['<sos>']] + r_idxs + [char2idx['<eos>']]
    # p_idxs = [char2idx['<sos>']] + p_idxs + [char2idx['<eos>']]
    r_tokens.append(r_idxs)
    p_tokens.append(p_idxs)

df['reactants_tokens'] = r_tokens
df['products_tokens'] = p_tokens
pd.to_pickle(df, 'processed_tokens.pickle')
