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
            torch.nn.ReLU(),
            torch.nn.Linear(4*self.n_embedding, self.n_embedding),
            torch.nn.Dropout(0.1)
        )
        
    def forward(self, x):

        return self.sequence(x)
    
class Block(torch.nn.Module):
    def __init__(self, n_heads, n_embedding, block_size):
        super(Block, self).__init__()
        self.n_heads = n_heads
        self.n_embedding = n_embedding
        self.block_size = block_size
        self.head_size = n_embedding // n_heads # the size of each head, because wee concatenate the heads
        self.norm1 = torch.nn.LayerNorm(self.n_embedding)
        self.norm2 = torch.nn.LayerNorm(self.n_embedding)
        self.attn = MultiHeadAttention(self.n_heads, self.head_size, self.n_embedding, self.block_size)
        self.ffn = FeedForward(self.n_embedding)

    def forward(self, x):
        x = x + self.attn(self.norm1(x)) # In GPT like models we use Pre-LayerNorm instead of Post-LayerNorm
        x = x + self.ffn(self.norm2(x))

        return x

class PositionalEncoding(torch.nn.Module):
    def __init__(self, block_size, n_embedding, device):
        super(PositionalEncoding, self).__init__()
        self.block_size = block_size
        self.n_embedding = n_embedding
        self.device = device
        self.pe = torch.nn.Embedding(self.block_size, self.n_embedding)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=self.device)

        return self.pe(positions).unsqueeze(0)


class LanguageModel(torch.nn.Module):
    def __init__(self, n_blocks, vocab_size, n_heads, n_embedding, block_size, device):
        super(LanguageModel, self).__init__()
        self.n_heads = n_heads
        self.n_embedding = n_embedding
        self.block_size = block_size
        self.n_blocks = n_blocks
        self.vocab_size = vocab_size
        self.device = device

        self.position_embedding = PositionalEncoding(self.block_size, self.n_embedding, self.device)
        self.tokens_embedding = torch.nn.Embedding(self.vocab_size, self.n_embedding)

        self.blocks = torch.nn.ModuleList([Block(self.n_heads, self.n_embedding, self.block_size) for _ in range(self.n_blocks)])
        self.linear = torch.nn.Linear(self.n_embedding, self.vocab_size)
        self.norm = torch.nn.LayerNorm(self.n_embedding)

    def forward(self, x):
        x = self.tokens_embedding(x) + self.position_embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.linear(x)

        return logits

    def generate(self, x, n_tokens, decoder):
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

