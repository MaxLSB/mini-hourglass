import torch
from model import LanguageModel

###############################################################

batch_size = 64
block_size = 256
epochs = 30
learning_rate = 3e-4
n_blocks=6
n_heads=6
n_embedding=384

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

model = LanguageModel(n_blocks=n_blocks, vocab_size=vocab_size, n_heads=n_heads, n_embedding=n_embedding, block_size=block_size, device=device).to(device)

model.load_state_dict(torch.load('model_assets/model.pth'))

context = input("Enter a prompt: ")

# Encode the context
context = torch.tensor(encoder(context), dtype=torch.long, device=device).unsqueeze(0)
# Generate text
print('Generated text: ', end='')
model.generate(context, n_tokens=100, decoder=decoder)

# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print('Generated text: ', end='')
# model.generate(context, n_tokens=100, decoder=decoder)


