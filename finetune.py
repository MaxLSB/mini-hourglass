import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import List
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from model import HourglassLM
from utils.tokenizer import Tokenizer

####################################################################################################

# Dataset class


class HaikuDataset(Dataset):
    def __init__(self, file_path, block_size, tokenizer):
        self.haikus = []
        self.block_size = block_size
        self.tokenizer = tokenizer

        print(f'Loading data from {file_path}...\n')
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Remove any potential trailing newlines or extra whitespace
        lines = [line.strip() for line in lines if line.strip()]

        # Iterate through the lines, taking 3 at a time
        for i in range(0, len(lines), 3):
            # Concatenate the 3 lines into one haiku, joined with '\n'
            haiku = '\n'.join(lines[i:i + 3])
            self.haikus.append(haiku)

    def __len__(self):
        return len(self.haikus)

    def __getitem__(self, idx):
        haiku = self.haikus[idx]
        encoded = self.tokenizer.encode_text(haiku)
        # Add EOS token at the end
        encoded = encoded + [self.tokenizer.eos]

        # Pad with EOS tokens or crop the sequence & adding the BOS token at the beginning
        if len(encoded) > self.block_size - 1:
            encoded = self.tokenizer.bos + encoded[:self.block_size - 1]
        else:
            encoded = self.tokenizer.bos + encoded + self.tokenizer.eos * (self.block_size - 1 - len(encoded))
        x = encoded[:-1]
        y = encoded[1:]

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


####################################################################################################

# Utility functions


def load_data(file_path: Path) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_model(filepath, device):
    checkpoint = torch.load(filepath, map_location=device)
    model = HourglassLM(
        vocab_size=checkpoint['hyperparameters']['vocab_size'],
        n_heads=checkpoint['hyperparameters']['n_heads'],
        n_embedding=checkpoint['hyperparameters']['n_embedding'],
        block_size=checkpoint['hyperparameters']['block_size'],
        dropout=checkpoint['hyperparameters']['dropout'],
        factors=checkpoint['hyperparameters']['factors']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model


def save_model(model, optimizer, epoch, loss, filepath):
    filepath = filepath + '.pth'
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
    print(f"Model saved at {filepath}")


####################################################################################################

# Main function


def main(args):

    # Initialize the custom character level tokenizer
    tokenizer = Tokenizer()
    tokenizer.load('vocab.json')  # Should be adjusted
    print(f'Vocabulary has been loaded.\n')

    # Load Pre-trained model
    model = load_model(Path(args.load_model_path),
                       args.device)  # Should be adjusted

    # Dataset and DataLoader
    dataset = HaikuDataset(args.data_path, args.block_size, tokenizer)
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Finetune model
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for x, y in tqdm(train_loader, desc="Fine-tuning"):
            x = x.to(args.device)
            y = y.to(args.device)

            y_pred = model(x)
            loss = loss_fn(y_pred.view(-1, model.vocab_size), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # Save finetuned model
    save_model(model, optimizer, epoch, total_loss, args.model_save_path)

    print('Finetuning has been completed successfully!')


if __name__ == "__main__":
    
    # Need to make the parser cleaner, in another file
    parser = argparse.ArgumentParser(
        description="Finetuning a pre-trained model")
    parser.add_argument("--load_model_path", type=str,
                        default='trained_models/hourglass-pretrained-9.pth', help="Path to pre-trained model")
    parser.add_argument("--data_path", type=str,
                        default='datasets/haiku-dataset.txt', help="Path to haiku dataset file")
    parser.add_argument("--model_save_path", type=str,
                        default='trained_models/hourglass-finetuned', help="Path to save finetuned model")
    parser.add_argument("--block_size", type=int,
                        default=256, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float,
                        default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs for finetuning")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available()
                        else "cpu", help="Device to use for training")

    args = parser.parse_args()
    main(args)
