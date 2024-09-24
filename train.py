import torch
from torch.utils.data import Dataset, DataLoader

###############################################################

batch_size = 64
block_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################

# Loading the data
with open('lovecraft-stories.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text))) # get a list of all unique characters in the dataset
vocab_size = len(chars)

char_to_idx = {ch: i for i, ch in enumerate(chars)} # character to index mapping
idx_to_char = {i: ch for i, ch in enumerate(chars)} # index to character mapping

def encoder(text):
    return [char_to_idx[ch] for ch in text]

def decoder(text):
    return ''.join([idx_to_char[i] for i in text])  

class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return (len(self.data) - self.block_size - 1) // self.block_size

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size
        x = self.data[start_idx:end_idx]
        y = self.data[start_idx + 1:end_idx + 1]
        return x, y

def train_val_split(data, split_ratio=0.95):
    n = int(split_ratio*len(data))
    return data[:n], data[n:]

data = torch.tensor(encoder(text), dtype=torch.long)
train, val = train_val_split(data)

train_data = TextDataset(train, block_size)
val_data = TextDataset(val, block_size)
train_loader = DataLoader(train_data, batch_size, shuffle=True)
test_loader = DataLoader(val_data, batch_size, shuffle=False)


