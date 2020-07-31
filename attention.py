import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, heads: int = 8):
        """
        Class for Multi-headed Self Attention module.

        :param embed_dim: Dimension of embedding vector
        :param heads: Number of attention heads
        """
        super().__init__()
        self.k, self.heads = embed_dim, heads
        """
        Each head has separate sets of three matrices, query, key and value weight 
        matrices. But it is more efficient to combine to a single k x (k*heads) matrices
        """
        self.tokeys = nn.Linear(embed_dim, embed_dim * heads, bias=False)
        self.toqueries = nn.Linear(embed_dim, embed_dim * heads, bias=False)
        self.tovalues = nn.Linear(embed_dim, embed_dim * heads, bias=False)

        # Reduce dimension back to k after concatenating outputs from multiple heads
        self.unifyheads = nn.Linear(heads * embed_dim, embed_dim)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads

        """
        Query - embedding vector
        Key - from memory (think of each key having associated values)
        Value - from memory
        """

        """
        The output of each linear module has size (b, t, h*k).
        We can reshape this to (b, t, h, k) to give each head its own dimension
        """

        queries = self.toqueries(x).view(b, t, h, k)
        keys = self.tokeys(x).view(b, t, h, k)
        values = self.tovalues(x).view(b, t, h, k)

        # Fold heads into batch dimensions
        keys = keys.transpose(1, 2).contiguous().view(b*h, t, k)
        queries = queries.transpose(1, 2).contiguous().view(b*h, t, k)
        values = values.transpose(1, 2).contiguous().view(b*h, t, k)

        """
        Scale queries and keys first
        """
        queries = queries / (k**(1/4))
        keys = keys / (k**(1/4))

        """
        Take the query, multiply with transposed key, take the softmax, this gives you a probability distribution over keys
        multiply with value. This is like an indexing scheme.
        """

        """
        dot product between queries and keys give an idea of how similar they are
        
        This will be high when the key and the query are very similar
        """
        dot = torch.bmm(queries, keys.transpose(1, 2))

        """
        To get probabilities of keys. The key with the biggest dot product will have the largest value.
        """
        dot = F.softmax(dot, dim=2)

        out = torch.bmm(dot, values).view(b, h, t, k)

        out = out.transpose(1, 2).contiguous().view(b, t, h * k)
        return self.unifyheads(out)
