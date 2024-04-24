import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from tqdm import tqdm
import numpy as np


with open('vocab.txt', 'r') as f:
    idx2char = f.read().splitlines()
char2idx = {k: v for v, k in enumerate(idx2char)}


class USPTO50(Dataset):
    def __init__(self, file_name: str='processed_tokens.pickle'):
        df = pd.read_pickle(file_name)
        self.reactants = df['reactants_tokens'].tolist()
        self.products = df['products_tokens'].tolist()

    def __len__(self):
        return len(self.reactants)

    def __getitem__(self, idx):
        r, p = self.reactants[idx], self.products[idx]
        p = [char2idx['<sos>']] + p
        r = r + [char2idx['<eos>']]
        return torch.tensor(p), torch.tensor(r)


def collate_fn(batch):
    reactants, products = zip(*batch)
    reactants = torch.nn.utils.rnn.pad_sequence(reactants, batch_first=True, padding_value=char2idx['<pad>'])
    products = torch.nn.utils.rnn.pad_sequence(products, batch_first=True, padding_value=char2idx['<pad>'])
    return reactants, products


dataset = USPTO50(file_name='processed_tokens.pickle')
x = np.load('train_indices.npy')
y = np.load('test_indices.npy')
z = np.load('val_indices.npy')
print(x.shape, y.shape, z.shape)
# x = np.arange(len(dataset))
# np.random.shuffle(x)
# a, b, c = 40030, 5003, 5004
# np.save('train_indices.npy', x[:a])
# np.save('test_indices.npy', x[a:a+b])
# np.save('val_indices.npy', x[a+b:])
# print(x.shape)
exit()

val_data = DataLoader(dataset, batch_size=512, collate_fn=collate_fn, num_workers=9)
print(f'Number of samples: {len(dataset)}')
print(f'Number of batches: {len(val_data)}')

r_sz, p_sz = 0, 0
for reactants, products in tqdm(val_data):
    if reactants.shape[1] > r_sz:
        r_sz = reactants.shape[1]
    if products.shape[1] > p_sz:
        p_sz = products.shape[1]

print(r_sz, p_sz)
