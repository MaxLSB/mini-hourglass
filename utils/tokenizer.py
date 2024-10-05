import json
import unicodedata
import re


####################################################################################################

# Tokenizer class

class Tokenizer:
    def __init__(self):
        self.chars = []
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0

    def fit(self, text):
        self.chars = sorted(list(set(text)))
        self.chars = ['<EOS>', '<UNK>'] + self.chars
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        print(f'Tokenizer has been fitted with {self.vocab_size} characters\n')

    def save(self, vocab_path):
        with open(vocab_path, 'w') as f:
            json.dump(self.char_to_idx, f)
        print(f'Vocabulary has been saved to {vocab_path}\n')

    def load(self, vocab_path):
        with open(vocab_path, 'r') as f:
            self.char_to_idx = json.load(f)
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        self.chars = list(self.char_to_idx.keys())
        self.vocab_size = len(self.chars)

    def encode_text(self, text):
        return [self.char_to_idx[ch] if ch in self.char_to_idx else self.char_to_idx['<UNK>'] for ch in text]

    def decode_text(self, indices):
        return ''.join([self.idx_to_char[i] for i in indices])


def clean_text(text):
    allowed_chars = set(
        "\n !&'$,-.:;?abcdefghijklmnopqrstuvwxyz")  # ABCDEFGHIJKLMNOPQRSTUVWXYZ
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    # text = re.sub(r'\s+', ' ', text)
    text = re.sub(rf"[^{''.join(allowed_chars)}]", "", text)

    return text
