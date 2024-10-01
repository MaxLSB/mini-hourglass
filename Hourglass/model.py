from typing import List
import torch
import math

class AttentionHead(torch.nn.Module):
    def __init__(self, head_size, n_embedding, block_size):
        super(AttentionHead, self).__init__()
        self.head_size = head_size
        self.block_size = block_size

        self.q = torch.nn.Linear(n_embedding, head_size, bias=False)
        self.k = torch.nn.Linear(n_embedding, head_size, bias=False)
        self.v = torch.nn.Linear(n_embedding, head_size, bias=False)
        self.dropout = torch.nn.Dropout(0.1) # dropout layer, same value as in the paper
        # Lower triangular matrix to mask the future tokens
        self.register_buffer('tril', torch.tril(torch.ones(self.block_size, self.block_size)))

    def forward(self, x):

        T = x.size(1) # the length of the sequence
        key = self.k(x) # the key of the attention mechanism
        query = self.q(x) # the query of the attention mechanism
        value = self.v(x) # the value of the attention mechanism

        output = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)
        output = output.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        output = torch.nn.functional.softmax(output, dim=-1)
        output = self.dropout(output)
        output = torch.matmul(output, value)

        return output

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_heads, head_size, n_embedding, block_size):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.n_embedding = n_embedding
        self.block_size = block_size
        self.head_size = head_size

        self.heads = torch.nn.ModuleList([AttentionHead(self.head_size, self.n_embedding, self.block_size) for _ in range(n_heads)])
        self.linear = torch.nn.Linear(self.n_heads*self.head_size, self.n_embedding)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.linear(output))

class FeedForward(torch.nn.Module):
    def __init__(self, n_embedding):
        super(FeedForward, self).__init__()
        self.n_embedding = n_embedding
        self.sequence = torch.nn.Sequential(
            torch.nn.Linear(self.n_embedding, 4*self.n_embedding),
            torch.nn.ReLU(), # Maybe use Squared ReLU as it has been shown to work better
            torch.nn.Linear(4*self.n_embedding, self.n_embedding),
            torch.nn.Dropout(0.1)
        )
        
    def forward(self, x):
        return self.sequence(x)
    
class TransformerLayer(torch.nn.Module):
    def __init__(self, n_heads, n_embedding, block_size):
        super(TransformerLayer, self).__init__()
        self.n_heads = n_heads
        self.n_embedding = n_embedding
        self.block_size = block_size
        self.head_size = n_embedding // n_heads # the size of each head, because wee concatenate the heads
        self.norm1 = torch.nn.LayerNorm(self.n_embedding)
        self.norm2 = torch.nn.LayerNorm(self.n_embedding)
        self.attn = MultiHeadAttention(self.n_heads, self.head_size, self.n_embedding, self.block_size)
        self.ffn = FeedForward(self.n_embedding)

    def forward(self, x):
        x = x + self.attn(self.norm1(x)) # In GPT like models the LayerNorm is applied before the main layers not after
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


class Hourglass(torch.nn.Module):
    """Hourglass Reccursive Block
    """
    def __init__(self, vocab_size, n_heads, n_embedding, block_size, factors: List[List[int]]): # Add dropout as a parameter
        super(Hourglass, self).__init__()
        self.n_heads = n_heads
        self.n_embedding = n_embedding
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.factors = factors
        self.n_layers = factors[0][0] # Number of layers in the block

        # Pre-Vanilla Transformer Decoder layers 
        self.pre_layer = torch.nn.ModuleList([TransformerLayer(self.n_heads, self.n_embedding, self.block_size) for _ in range(self.n_layers)])

        if len(self.factors) == 1:
            # We are at the last layer
            self.hourglass = None
        else:
            self.k = factors[1][1] # Factor for the linear pooling
            self.linearProjection = torch.nn.Linear(self.k * self.n_embedding, self.n_embedding) # For Linear Pooling
            self.hourglass = Hourglass(vocab_size, n_heads, n_embedding, block_size, factors[1:]) # We go to the next tuple in the hierarchy
            # Post-Vanilla Transformer Decoder layers
            self.post_layer = torch.nn.ModuleList([TransformerLayer(self.n_heads, self.n_embedding, self.block_size) for _ in range(self.n_layers)])

    def forward(self, x):
        for i in range(self.n_layers):
            x = self.pre_layer[i](x) # Pre-Vanilla Transformer Layer

        if self.hourglass is not None:
            x_hourglass = self.shiftRight(x) # Shift the sequence to the right by k-1 positions
            x_hourglass = self.linearPooling(x_hourglass)
            x_hourglass = self.hourglass(x_hourglass)
            x_hourglass = self.upsampling(x_hourglass)
            x = x + x_hourglass # Residual connection
            for i in range(self.n_layers):
                x = self.post_layer[i](x) # Post-Vanilla Transformer Layer, if we are not at the last layer

        return x
    
    def shiftRight(self, x): # Correct ShiftRight
        """Shift the sequence to the right by k-1 positions
        """
        B, T, C = x.size()
        shifted_x = torch.zeros(B, T, C, device=x.device)
        shifted_x[:, self.k-1:T, :] = x
        return shifted_x

    def linearPooling(self, x):
        """Shortening (Downsampling) the sequence length by a factor of k
        - Add padding if T is not divisible by k ?
        """
        B, T, C = x.size()
        x = x.reshape(B, T // self.k, self.k*C) # (B, T//k, k*C)
        return self.linearProjection(x) # (B, T//k, C)
    
    def upsampling(self, x):
        """Upsampling the sequence length by a factor of k (simplest method)
        """
        return torch.repeat_interleave(x, repeats=self.k, dim=1) # (B, k*T, C)


class HourglassLM(torch.nn.Module):
    """Hourglass Language Model
    Note: Add dropout as a parameter ?
    """
    def __init__(self, vocab_size, n_heads, n_embedding, block_size, factors: List[List[int]] ):
        super(HourglassLM, self).__init__()
        self.n_heads = n_heads
        self.n_embedding = n_embedding
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.factors = factors

        self.position_embedding = PositionalEncoding(self.block_size, self.n_embedding)
        self.tokens_embedding = torch.nn.Embedding(self.vocab_size, self.n_embedding)

        # Layers for the final output
        self.linear = torch.nn.Linear(self.n_embedding, self.vocab_size)
        self.norm = torch.nn.LayerNorm(self.n_embedding)

        self.hourglass = Hourglass(self.vocab_size, self.n_heads, self.n_embedding, self.block_size, self.factors)

    def forward(self, x):
        x = self.tokens_embedding(x) + self.position_embedding(x)  # Problem car cela initialise les embeddings a chaque itération

        x = self.hourglass(x) # We pass the input through the hourglass reccursive block

        x = self.norm(x)
        logits = self.linear(x)

        return logits # We return the output of the model
    

    def generate(self, x, n_tokens, decoder):
        """Generate a text of n_tokens given a prompt x
        """
        for _ in range(n_tokens):
            context = x[:, -self.block_size:]  # crop the context to the block size
            logits = self.forward(context)  # (B, T, vocab_size), we output the logits
            logit = logits[:, -1]  # (B, vocab_size), logit of the last token
            probs = torch.nn.functional.softmax(logit, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1), we sample the next token

            # We print the predicted token
            token_id = next_token.item()  
            print(decoder([token_id]), end='', flush=True)

            x = torch.cat((x, next_token), dim=1)  # (B, T+1), we add the new token to the context

        return x