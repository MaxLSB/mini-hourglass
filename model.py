import torch
import torch.functional as F

class AttentionHead(torch.nn.Module):
    def __init__(self, head_size, n_embedding, block_size):
        super(AttentionHead, self).__init__()
        self.head_size = head_size
        self.block_size = block_size

        self.q = torch.nn.Linear(n_embedding, head_size, biais=False)
        self.k = torch.nn.Linear(n_embedding, head_size, biais=False)
        self.v = torch.nn.Linear(n_embedding, head_size, biais=False)
        self.dropout = torch.nn.Dropout(0.1) # dropout layer, same value as in the paper
        # Lower triangular matrix to mask the future tokens
        self.register_buffer('tril', torch.tril(torch.ones(self.block_size, self.block_size)))

    def __forward__(self, x):

        B,T,C = x.shape
        key = self.k(x) # the key of the attention mechanism
        query = self.q(x) # the query of the attention mechanism
        value = self.v(x) # the value of the attention mechanism

        output = torch.matmul(query, key.T) / torch.sqrt(self.head_size)
        output = output.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        output = F.softmax(output, dim=-1)
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

class Block(torch.nn.Module):
    def __init__(self, n_heads, n_embedding, block_size):
        super(Block, self).__init__()
        self.n_heads = n_heads
        self.n_embedding = n_embedding
        self.block_size = block_size
        self.head_size = n_embedding // n_heads # the size of each head, because wee concatenate the heads

    def forward(self, x):
        pass

class FeedForward(torch.nn.Module):
    def __init__(self, n_embedding):
        self.n_embedding = n_embedding
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(self.n_embedding, 4*self.n_embedding),
            torch.nn.ReLU(),
            torch.nn.Linear(4*self.n_embedding, self.n_embedding),
            torch.nn.Dropout(0.1)
        )
        
    def forward(self, x):
        return self.ffn(x)