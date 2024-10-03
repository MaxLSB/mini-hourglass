import torch
import time
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import HourglassLM
from utils.config import get_parser

####################################################################################################

# Dataset class


class TextDataset(Dataset):  # Dataset class for text data
    def __init__(self, data: torch.Tensor, block_size: int):
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

####################################################################################################

# Utility functions


def load_and_preprocess_data(file_path: Path) -> Tuple[List[str], dict, dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Get all unique characters (like a Tokenizer)
    chars = sorted(list(set(text)))
    # Create a dictionary mapping characters to indices
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    # Create a dictionary mapping indices to characters
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return chars, char_to_idx, idx_to_char


# Encode text to a list of indices
def encode_text(text: str, char_to_idx: dict) -> List[int]:
    return [char_to_idx[ch] for ch in text]


# Decode indices to text
def decode_text(indices: List[int], idx_to_char: dict) -> str:
    return ''.join([idx_to_char[i] for i in indices])


def train_val_split(data: torch.Tensor, split_ratio: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    n = int(split_ratio * len(data))
    return data[:n], data[n:]


# Initialize weights using Xavier initialization
def init_weights(module):
    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
        torch.nn.init.xavier_normal_(module.weight)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)  # Initialize biases to zero
    elif isinstance(module, torch.nn.LayerNorm):
        # Initialize LayerNorm layers to unity
        torch.nn.init.ones_(module.weight)
        torch.nn.init.zeros_(module.bias)


def evaluate(model: HourglassLM, val_loader: DataLoader, loss_fn: torch.nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            total_loss += loss_fn(y_pred.view(-1,
                                  model.vocab_size), y.view(-1)).item()
    return total_loss / len(val_loader)


def save_model(model, optimizer, epoch, loss, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss,
                'hyperparameters': {
        'vocab_size': model.vocab_size,
        'n_heads': model.n_heads,
        'n_embedding': model.n_embedding,
        'block_size': model.block_size,
        'dropout': model.dropout,
        'factors': model.factors}}, Path(filepath))

####################################################################################################

# Main function


def main():
    parser = get_parser()
    args = parser.parse_args()

    chars, char_to_idx, idx_to_char = load_and_preprocess_data(
        Path(args.data_path))
    special_tokens = ['<PAD>', '<EOS>']
    # Add special tokens at position 0 and 1 for the fine-tuning phase
    chars = special_tokens + chars
    vocab_size = len(chars)

    # Load and preprocess data
    data = torch.tensor(encode_text(open(
        Path(args.data_path), 'r', encoding='utf-8').read(), char_to_idx), dtype=torch.long)
    train_data, val_data = train_val_split(data, args.train_val_split)
    train_dataset = TextDataset(train_data, args.batch_size)
    val_dataset = TextDataset(val_data, args.batch_size)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)

    # Model and Initialization
    model = HourglassLM(vocab_size=vocab_size, n_heads=args.n_heads,
                        n_embedding=args.n_embedding, block_size=args.block_size,
                        dropout=args.dropout, factors=args.factors).to(args.device)

    # Results are not good for now with this initialization
    # model.apply(init_weights)

    # Optimizer and Loss function
    optimizer = torch.optim.AdamW(model.parameters(
    ), lr=args.learning_rate, betas=args.betas, eps=args.eps)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Model information
    print(f'Parameters: {sum(p.numel()
          for p in model.parameters()) / 1e6:.2f}M')
    print(f'Model hyperparameters | vocab_size: {vocab_size}, block_size: {args.block_size}, batch_size: {args.batch_size}, '
          f'n_heads: {args.n_heads}, n_embedding: {args.n_embedding}, factors: {args.factors}')
    print(f'Number of characters in the training dataset: {
          len(train_data) / 1e6:.2f}M')

    start_time = time.time()

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for x, y in tqdm(train_loader, desc="Training"):
            x, y = x.to(args.device), y.to(args.device)
            y_pred = model(x)
            loss = loss_fn(y_pred.view(-1, model.vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(args.device), y.to(args.device)
                y_pred = model(x)
                total_loss += loss_fn(y_pred.view(-1,
                                      model.vocab_size), y.view(-1)).item()
        val_loss = total_loss / len(val_loader)
        print(f'Epoch: {epoch+1}/{args.epochs}, Training Loss: {
              train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    end_time = time.time()
    print(f'Training time: {end_time - start_time:.2f} seconds')

    # Saving the model
    save_model(model, optimizer, epoch, train_loss, Path(args.model_save_path))
    print(f'Model saved to {args.model_save_path}')


if __name__ == "__main__":
    main()
