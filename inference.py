import torch
from pathlib import Path
from typing import List, Tuple
from typing import List, Callable
from model import HourglassLM
from utils.config import get_parser


def load_and_preprocess_data(file_path: Path) -> Tuple[List[str], dict, dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)  # Get the number of unique characters
    # Create a dictionary mapping characters to indices
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    # Create a dictionary mapping indices to characters
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return vocab_size, char_to_idx, idx_to_char


# Encode text to a list of indices
def encoder(text: str, char_to_idx: dict) -> List[int]:
    return [char_to_idx[ch] for ch in text]


# Decode indices to text
def decoder(indices: List[int], idx_to_char: dict) -> str:
    return ''.join([idx_to_char[i] for i in indices])


def generate_from_prompt(model: HourglassLM, encoder: Callable[[str], List[int]],
                         decoder: Callable[[List[int]], str], prompt: str,
                         n_tokens: int, device: torch.device) -> str:
    context = torch.tensor(
        encoder(prompt), dtype=torch.long, device=device).unsqueeze(0)
    generated_indices = model.generate(context, n_tokens=n_tokens)
    return decoder(generated_indices[0].tolist())


def generate_from_scratch(model: HourglassLM, decoder: Callable[[List[int]], str],
                          n_tokens: int, device: torch.device) -> str:
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_indices = model.generate(context, n_tokens=n_tokens)
    return decoder(generated_indices[0].tolist())


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Vocabulary size, character to index and index to character mappings
    vocab_size, char_to_idx, idx_to_char = load_and_preprocess_data(
        Path(args.data_path))

    # Create the model and load the trained weights
    model = HourglassLM(vocab_size=vocab_size, n_heads=args.n_heads,
                        n_embedding=args.n_embedding, block_size=args.block_size, dropout=args.dropout, factors=args.factors).to(args.device)
    model.load_state_dict(torch.load(Path(args.model_save_path)))
    model.eval()

    # Generate using the prompt as context
    if args.gen_mode == 'prompt':
        prompt = input("Enter a prompt: ")
        prompt = torch.tensor(
            encoder(prompt, char_to_idx), dtype=torch.long, device=args.device).unsqueeze(0)
        print('Generated text: ', end='')
        model.generate(context, args.max_tokens,
                       decoder=decoder, idx_to_char=idx_to_char)

    # Generate from scratch
    elif args.gen_mode == 'scratch':
        context = torch.zeros((1, 1), dtype=torch.long, device=args.device)
        print('Generated text: ', end='')
        model.generate(context, args.max_tokens,
                       decoder=decoder, idx_to_char=idx_to_char)


if __name__ == "__main__":
    main()
