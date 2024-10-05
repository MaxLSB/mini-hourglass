import torch
import random
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader

from model import HourglassLM
from utils.config import get_parser
from utils.tokenizer import Tokenizer, clean_text

####################################################################################################

# Dataset class


# Random sampling Dataset to learn different patterns
class RandomSampleTextDataset(Dataset):
    def __init__(self, text, block_size, tokenizer, batch_size, iterations=1000):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.iterations = iterations
        self.batch_size = batch_size
        self.data = tokenizer.encode_text(text)

    def __len__(self):
        # Number of batches is the number of iterations
        return self.iterations*self.batch_size

    def __getitem__(self, idx):
        start_idx = random.randint(
            0, len(self.data) - self.block_size - 1)  # Random start index
        x = self.data[start_idx:start_idx + self.block_size]
        y = self.data[start_idx + 1:start_idx + self.block_size + 1]
        # y = x[1:] + [self.tokenizer['<EOS>'] # Need to test this.
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


####################################################################################################

# Utility functions


def load_data(file_path: Path) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def train_val_split(data: torch.Tensor, split_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
    n = int(split_ratio * len(data))
    return data[:n], data[n:]


# Initialize weights using Xavier initialization
def init_weights(module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


def save_model(model, optimizer, epoch, loss, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    epoch_filepath = filepath + f'-{epoch+1}.pth'
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
        'factors': model.factors}}, Path(epoch_filepath))
    print(f'Model saved to {epoch_filepath}\n')


####################################################################################################

# Main function


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Load the data
    print(f'Loading data from {args.data_path}...\n')
    text = load_data(Path(args.data_path))
    print(f'Cleaning the data...\n')
    text = clean_text(text)

    # Initialize the custom character level tokenizer
    print('Initializing the tokenizer...\n')
    tokenizer = Tokenizer()
    tokenizer.fit(text)
    tokenizer.save(args.vocab_path)  # Save the tokenizer to use it later

    # Split and Preprocess data
    train_data, val_data = train_val_split(text, args.train_val_split)
    del text  # Free up memory
    train_dataset = RandomSampleTextDataset(
        train_data, block_size=args.block_size, tokenizer=tokenizer, batch_size=args.batch_size, iterations=args.iter)
    val_dataset = RandomSampleTextDataset(
        val_data, block_size=args.block_size, tokenizer=tokenizer, batch_size=args.batch_size, iterations=args.eval_iter)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)

    # Model and Initialization
    model = HourglassLM(vocab_size=tokenizer.vocab_size, n_heads=args.n_heads,
                        n_embedding=args.n_embedding, block_size=args.block_size,
                        dropout=args.dropout, factors=args.factors).to(args.device)

    # Results are not good for now with this initialization
    model.apply(init_weights)

    # Optimizer and Loss function
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=args.betas, eps=args.eps)

    loss_fn = torch.nn.CrossEntropyLoss()

    # Model information
    print(f'Parameters: {sum(p.numel()
          for p in model.parameters()) / 1e6:.2f}M')
    print(f'Model hyperparameters | vocab_size: {tokenizer.vocab_size}, block_size: {args.block_size}, batch_size: {args.batch_size}, '
          f'n_heads: {args.n_heads}, n_embedding: {args.n_embedding}, factors: {args.factors}')
    print(f'Number of characters in the training dataset: {
          len(train_data) / 1e6:.2f}M')

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for x, y in tqdm(train_loader, desc="Training"):
            x = x.to(args.device)
            y = y.to(args.device)

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
                x = x.to(args.device)
                y = y.to(args.device)
                y_pred = model(x)
                total_loss += loss_fn(y_pred.view(-1,
                                      model.vocab_size), y.view(-1)).item()
        val_loss = total_loss / len(val_loader)

        # Print the losses
        print(f'Epoch: {epoch+1}/{args.epochs}, Training Loss: {
              train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        # Save the model at each Epoch
        save_model(model, optimizer, epoch, train_loss,
                   args.model_save_path)

    print('Training has been completed successfully!')


if __name__ == "__main__":
    main()
