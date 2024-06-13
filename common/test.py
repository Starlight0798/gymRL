import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"

        self.head_dim = embed_dim // num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        self.out_linear = nn.Linear(embed_dim, embed_dim)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Split and reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        out = self.out_linear(context)
        return out, attn_weights

# Example usage
embed_dim = 512
num_heads = 8
dropout = 0.1

multi_head_attn = MultiHeadAttention(embed_dim, num_heads, dropout)

# Example input: (batch_size, feature_dims)
x = torch.randn(64, embed_dim)  # Example input (batch_size, feature_dims)

# Convert 2D input to 3D
x = x.unsqueeze(1)  # (batch_size, 1, embed_dim)

# Using the same tensor for query, key, and value in this example
query = x
key = x
value = x
mask = None  # Or some mask if needed

output, attn_weights = multi_head_attn(query, key, value, mask)

# If needed, convert the 3D output back to 2D
output = output.squeeze(1)

print("Output shape:", output.shape)
print("Attention weights shape:", attn_weights.shape)
