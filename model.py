from typing import List
import torch
import math

####################################################################################################

# Components of the Transformer


class AttentionHead(torch.nn.Module):
    def __init__(self, head_size, n_embedding, block_size, dropout):
        super(AttentionHead, self).__init__()
        self.head_size = head_size
        self.block_size = block_size
        self.dropout = dropout

        self.q = torch.nn.Linear(n_embedding, head_size, bias=False)
        self.k = torch.nn.Linear(n_embedding, head_size, bias=False)
        self.v = torch.nn.Linear(n_embedding, head_size, bias=False)
        self.dropout = torch.nn.Dropout(self.dropout)
        # Lower triangular matrix to mask the future tokens
        self.register_buffer('tril', torch.tril(
            torch.ones(self.block_size, self.block_size)))

    def forward(self, x):

        T = x.size(1)  # length of the sequence
        key = self.k(x)  # key of the attention mechanism
        query = self.q(x)  # query of the attention mechanism
        value = self.v(x)  # value of the attention mechanism

        output = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(self.head_size)
        output = output.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        output = torch.nn.functional.softmax(output, dim=-1)
        output = self.dropout(output)
        output = torch.matmul(output, value)

        return output


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_heads, head_size, n_embedding, block_size, dropout):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.n_embedding = n_embedding
        self.block_size = block_size
        self.head_size = head_size
        self.dropout = dropout

        self.heads = torch.nn.ModuleList([AttentionHead(
            self.head_size, self.n_embedding, self.block_size, self.dropout) for _ in range(n_heads)])
        self.linear = torch.nn.Linear(
            self.n_heads*self.head_size, self.n_embedding)
        self.dropout = torch.nn.Dropout(self.dropout)

    def forward(self, x):
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.linear(output))


class FeedForward(torch.nn.Module):
    def __init__(self, n_embedding, dropout):
        super(FeedForward, self).__init__()
        self.n_embedding = n_embedding
        self.dropout = dropout
        self.sequence = torch.nn.Sequential(
            torch.nn.Linear(self.n_embedding, 4*self.n_embedding),
            torch.nn.ReLU(),  # Maybe use Squared ReLU as it has been shown to work better
            torch.nn.Linear(4*self.n_embedding, self.n_embedding),
            torch.nn.Dropout(self.dropout)
        )

    def forward(self, x):
        return self.sequence(x)


class TransformerLayer(torch.nn.Module):
    def __init__(self, n_heads, n_embedding, block_size, dropout):
        super(TransformerLayer, self).__init__()
        self.n_heads = n_heads
        self.n_embedding = n_embedding
        self.block_size = block_size
        self.dropout = dropout
        # the size of each head, because wee concatenate the heads
        self.head_size = n_embedding // n_heads
        self.norm1 = torch.nn.LayerNorm(self.n_embedding)
        self.norm2 = torch.nn.LayerNorm(self.n_embedding)
        self.attn = MultiHeadAttention(
            self.n_heads, self.head_size, self.n_embedding, self.block_size, self.dropout)
        self.ffn = FeedForward(self.n_embedding, self.dropout)

    def forward(self, x):
        # In GPT like models the LayerNorm is applied before the main layers not after
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class PositionalEncoding(torch.nn.Module):
    def __init__(self, block_size, n_embedding):
        super(PositionalEncoding, self).__init__()
        self.block_size = block_size
        self.n_embedding = n_embedding
        self.pe = torch.nn.Embedding(self.block_size, self.n_embedding)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return self.pe(positions).unsqueeze(0)

####################################################################################################

# Hourglass: Hierarchical Transformer


class Hourglass(torch.nn.Module):
    """Hourglass Reccursive Block
    """
    # Add dropout as a parameter

    def __init__(self, vocab_size, n_heads, n_embedding, block_size, dropout, factors: List[int]):
        super(Hourglass, self).__init__()
        self.n_heads = n_heads
        self.n_embedding = n_embedding
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.factors = factors
        self.n_layers = factors[0]  # Number of layers in the current Hourglass

        # Pre-Vanilla Transformer Decoder layers
        self.pre_layer = torch.nn.ModuleList([TransformerLayer(
            self.n_heads, self.n_embedding, self.block_size, self.dropout) for _ in range(self.n_layers)])

        if len(self.factors) == 2:
            # We are at the last layer, so the last pair of elements in the factors list
            self.hourglass = None
        else:
            self.k = factors[2]  # Factor for the linear pooling
            self.linearProjection = torch.nn.Linear(
                self.k * self.n_embedding, self.n_embedding)  # For Linear Pooling
            # We go to the next tuple in the hierarchy
            self.hourglass = Hourglass(
                self.vocab_size, self.n_heads, self.n_embedding, self.block_size, self.dropout, self.factors[2:])
            # Post-Vanilla Transformer Decoder layers
            self.post_layer = torch.nn.ModuleList([TransformerLayer(
                self.n_heads, self.n_embedding, self.block_size, self.dropout) for _ in range(self.n_layers)])

    def forward(self, x):
        T = x.size(1)  # the length of the sequence
        for i in range(self.n_layers):
            x = self.pre_layer[i](x)  # Pre-Vanilla Transformer Layer
        if self.hourglass is not None:
            # Shift the sequence to the right by k-1 positions so that the information does not leak
            x_hourglass = self.shiftRight(x)
            x_hourglass = self.linearPooling(x_hourglass)
            x_hourglass = self.hourglass(x_hourglass)
            x_hourglass = self.upsampling(x_hourglass, T)
            # Residual connection. But since x_hourglass is shifted, does it make sense to add them ?
            x = x + x_hourglass
            for i in range(self.n_layers):
                # Post-Vanilla Transformer Layer, if we are not at the last layer
                x = self.post_layer[i](x)

        return x

    def shiftRight(self, x):  # Correct ShiftRight
        """Shift the sequence to the right by k-1 positions
        """
        B, T, C = x.size()
        shifted_x = torch.zeros(B, self.k, C, device=x.device)  # (B, k, C)
        shifted_x = torch.concatenate(
            (shifted_x, x[:, :T-self.k+1]), dim=1)  # (B, k+(T-k), C)
        return shifted_x

    def linearPooling(self, x):
        """Shortening (Downsampling) the sequence length by a factor of k
        """
        if x.size(1) % self.k != 0:  # Padding if T is not divisible by k
            # Number of elements to pad
            pad_size = self.k - (x.size(1) % self.k)
            padding = (0, 0, 0, pad_size)  # (left, right, top, bottom)
            x = torch.nn.functional.pad(x, padding)

        x = x.reshape(x.size(0), x.size(1) // self.k,
                      self.k*x.size(2))  # (B, T//k, k*C)
        return self.linearProjection(x)  # (B, T//k, C)

    def upsampling(self, x, T):
        """Upsampling the sequence length by a factor of k (simplest method)
        """
        x = torch.repeat_interleave(x, repeats=self.k, dim=1)  # (B, k*T, C)
        return x[:, :T]  # If the sequence was padded, we remove the padding


class HourglassLM(torch.nn.Module):
    """Hourglass Language Model
    Note: Add dropout as a parameter ?
    """

    def __init__(self, vocab_size, n_heads, n_embedding, block_size, dropout, factors: List[int]):
        super(HourglassLM, self).__init__()
        self.n_heads = n_heads
        self.n_embedding = n_embedding
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.factors = factors
        self.dropout = dropout

        self.position_embedding = PositionalEncoding(
            self.block_size, self.n_embedding)
        self.tokens_embedding = torch.nn.Embedding(
            self.vocab_size, self.n_embedding)

        # Layers for the final output
        self.linear = torch.nn.Linear(self.n_embedding, self.vocab_size)
        self.norm = torch.nn.LayerNorm(self.n_embedding)

        self.hourglass = Hourglass(
            self.vocab_size, self.n_heads, self.n_embedding, self.block_size, self.dropout, self.factors)

    def forward(self, x):
        # Problem car cela initialise les embeddings a chaque itération
        x = self.tokens_embedding(x) + self.position_embedding(x)

        # We pass the input through the hourglass reccursive block
        x = self.hourglass(x)

        x = self.norm(x)
        logits = self.linear(x)

        return logits  # We return the output of the model

    def generate(self, x, max_tokens, decode_text, idx_to_char):
        """Generate a text of max_tokens given a prompt
        """
        for _ in range(max_tokens):
            # crop the context to the block size
            context = x[:, -self.block_size:]
            # (B, T, vocab_size), we output the logits
            logits = self.forward(context)
            logit = logits[:, -1]  # (B, vocab_size), logit of the last token
            probs = torch.nn.functional.softmax(logit, dim=-1)
            # (B, 1), we sample the next token
            next_token = torch.multinomial(probs, num_samples=1)

            # We print the predicted token
            token_id = next_token.item()
            print(decode_text([token_id], idx_to_char), end='', flush=True)

            # (B, T+1), we add the new token to the context
            x = torch.cat((x, next_token), dim=1)

        return x
