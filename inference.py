import torch
from pathlib import Path
from typing import List, Tuple
from typing import List, Callable

from utils.config import get_parser
from utils.tokenizer import Tokenizer
from model import HourglassLM

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


def encode_text(text: str, char_to_idx: dict) -> List[int]:
    return [char_to_idx[ch] if ch in char_to_idx else char_to_idx['<UNK>'] for ch in text]


def decode_text(indices: List[int], idx_to_char: dict) -> str:
    return ''.join([idx_to_char[i] for i in indices])


####################################################################################################

# Main function


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Initialize the custom character level tokenizer
    tokenizer = Tokenizer()
    tokenizer.load(args.vocab_path)

    # Create the model and load the trained weights
    model = load_model(Path(args.load_model_path), args.device)
    model.eval()

    # Generate using the prompt as context
    if args.gen_mode == 'prompt':
        prompt = input("Enter a prompt: ")
        prompt = torch.tensor(
            tokenizer.encode_text(prompt), dtype=torch.long, device=args.device).unsqueeze(0)
        print('\nGenerating text: \n')
        model.generate(prompt, max_tokens=args.max_tokens,
                       tokenizer=tokenizer)

    # Generate from scratch
    elif args.gen_mode == 'scratch':
        context = torch.zeros((1, 1), dtype=torch.long, device=args.device)
        print('\nGenerating text: \n')
        model.generate(context, max_tokens=args.max_tokens,
                       tokenizer=tokenizer)


if __name__ == "__main__":
    main()
