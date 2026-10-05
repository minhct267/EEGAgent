"""Rotary-attention blocks shared by the local EEG classifiers."""
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

def Rope(x):
    """Rotate each even/odd feature pair by a position-dependent angle (RoPE)."""
    B, T, C = x.shape
    theta = 10000 ** (-torch.arange(0, C, 2, device=x.device) / C)  # Inverse frequency per pair, shape (C // 2,).
    pos = torch.arange(T, device=x.device).unsqueeze(1)  # (T, 1)
    angles = pos * theta  # Angle at each time and frequency, shape (T, C // 2).
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    x1, x2 = x[:,:,::2], x[:,:,1::2]  # Even and odd features, each (B, T, C // 2).
    rotated_x1 = x1*cos-x2*sin  # (B, T, C // 2)
    rotated_x2 = x1*sin+x2*cos
    return torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)  # Interleave pairs back to (B, T, C).

class MultiHeadSelfAttentionRoPE(nn.Module):
    """Multi-head self-attention with rotary embeddings on queries and keys."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj.SCALE = 1  # Marks this projection so weight init uses the residual scale.

    def forward(self, x, pad_mask=None):
        # x is (batch, seq_len, embed_dim).
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, T, head_dim)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE expects (batch, time, dim), so fold heads into the batch.
        q = q.reshape(B * self.num_heads, T, self.head_dim)
        k = k.reshape(B * self.num_heads, T, self.head_dim)

        q = Rope(q)
        k = Rope(k)

        q = q.view(B, self.num_heads, T, self.head_dim)  # Restore (B, heads, T, head_dim).
        k = k.view(B, self.num_heads, T, self.head_dim)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, heads, T, T)
        if pad_mask is not None:  # pad_mask is (B, T); 0 marks positions to ignore.
            attn_scores = attn_scores.masked_fill(pad_mask[:, None, None, :] == 0, float("-inf"))
        attn_probs = F.softmax(attn_scores, dim=-1)  # (B, heads, T, T)

        out = torch.matmul(attn_probs, v)  # (B, heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, self.embed_dim)  # (B, T, embed_dim)


        out = self.out_proj(out)  # (B, T, embed_dim)
        return out, attn_probs
    
class FFN(nn.Module):
    """Position-wise feed-forward block used inside each transformer layer."""
    def __init__(self, embed_dim, drop_p=0.1) -> None:
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, 4 * embed_dim)
        self.GELU = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(embed_dim * 4, embed_dim)
        self.c_proj.SCALE = 1  # Marks this projection so weight init uses the residual scale.
        self.drop = nn.Dropout(drop_p)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.GELU(x)
        x = self.c_proj(x)
        x = self.drop(x)
        return x
    
class RoPETransformer(nn.Module):
    """Pre-norm block: rotary attention, then the feed-forward layer, each with a residual."""
    def __init__(self, embed_dim, num_heads) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.RoPEAttention = MultiHeadSelfAttentionRoPE(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim)
    
    def forward(self, x, mask=None):
        x1 = self.ln1(x)
        x1, attn = self.RoPEAttention(x1, pad_mask=mask)
        x = x + x1
        x1 = self.ln2(x)
        x = x + self.ffn(x1)
        return x, attn