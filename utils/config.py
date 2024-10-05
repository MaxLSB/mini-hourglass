import torch
import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model architecture

    parser.add_argument(
        "--factors",
        nargs='+',
        type=int,
        default=[6, 1],
        help="""
        List of Tuples defining the number of layers and the shortening factor for each block ofs the model. Example: 2 1 4 4
        """
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training."
    )

    parser.add_argument(
        "--block_size",
        type=int,
        default=256,
        help="Block size for training."
    )

    parser.add_argument(
        "--n_heads",
        type=int,
        default=6,
        help="Number of attention heads."
    )

    parser.add_argument(
        "--n_embedding",
        type=int,
        default=512,
        help="Embedding dimension."
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate."
    )

    # Training configuration
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs for training."
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=1000,
        help="Number of batches per epoch."
    )

    parser.add_argument(
        "--eval_iter",
        type=int,
        default=200,
        help="Number of batches per validation."
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=4e-4,
        help="Learning rate for training."
    )

    parser.add_argument(
        "--betas",
        nargs="+",
        type=float,
        default=[0.9, 0.98],
        help="Beta parameters for Adam optimizer."
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=1e-9,
        help="Epsilon parameter for Adam optimizer."
    )

    parser.add_argument(
        "--train_val_split",
        type=float,
        default=0.90,
        help="Ratio of training and validation data."
    )

    # Other configurations
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training (cuda/cpu)."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default='datasets/books_corpus.txt',
        help="Path to the data file."
    )

    parser.add_argument(
        "--model_save_path",
        type=str,
        default='trained_models/hourglass-pretrained',
        help="Path to save the model."
    )

    parser.add_argument(
        "--vocab_path",
        type=str,
        default='vocab.json',
        help="Path to save the vocabulary."
    )

    # Inference settings

    parser.add_argument(
        "--gen_mode",
        type=str.lower,
        choices=["prompt", "scratch"],
        default="scratch",
        help="Type of generation to perform, choose from prompt or random."
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=250,
        help="Maximum number of tokens to generate."
    )

    parser.add_argument(
        "--load_model_path",
        type=str,
        default='trained_models/hourglass-finetuned.pth',
        help="Load a trained model for inference."
    )

    return parser
