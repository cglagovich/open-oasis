"""
MLX implementation of axial attention modules
"""

from typing import Optional
import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from rotary_embedding_mlx import apply_rotary_emb

def scaled_dot_product_attention(query, key, value, is_causal=False):
    d_k = query.shape[-1]
    transpose_dims = tuple(range(query.ndim - 2)) + (query.ndim - 1, query.ndim - 2)
    scores = mx.matmul(query, key.transpose(*transpose_dims)) / mx.sqrt(d_k)
    
    if is_causal:
        mask = mx.triu(mx.ones((scores.shape[-2], scores.shape[-1])), k=1)
        scores = mx.where(mask == 0, scores, float('-inf'))
    
    attention_weights = mx.softmax(scores, axis=-1)
    return mx.matmul(attention_weights, value)

class TemporalAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: object,  # Replace with MLX RotaryEmbedding when available
        is_causal: bool = True,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)
        
        self.rotary_emb = rotary_emb
        self.is_causal = is_causal

    def __call__(self, x, kv_cache=None):
        B, T, H, W, D = x.shape

        # Generate Q, K, V
        qkv = self.to_qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        
        # Rearrange for temporal attention
        q = q.reshape(B, T, H, W, self.heads, self.head_dim).transpose(0,2,3,4,1,5).reshape(B*H*W, self.heads, T, self.head_dim)
        k = k.reshape(B, T, H, W, self.heads, self.head_dim).transpose(0,2,3,4,1,5).reshape(B*H*W, self.heads, T, self.head_dim)
        v = v.reshape(B, T, H, W, self.heads, self.head_dim).transpose(0,2,3,4,1,5).reshape(B*H*W, self.heads, T, self.head_dim)

        if kv_cache is not None:
            # I think rotaty embedding was getting messed up by q, k len 1
            cache_len = kv_cache[0].shape[2]
            q = mx.repeat(q, repeats=cache_len+1, axis=2)
            k = mx.repeat(k, repeats=cache_len+1, axis=2)
        # Apply rotary embeddings (implementation would depend on MLX version)
        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q, self.rotary_emb.freqs)
            k = self.rotary_emb.rotate_queries_or_keys(k, self.rotary_emb.freqs)

        if kv_cache is not None:
            q = q[:,:,-1:,...]
            k = k[:,:,-1:,...]
            k = mx.concatenate([kv_cache[0], k], axis=2)
            v = mx.concatenate([kv_cache[1], v], axis=2)
            # print(f"{k.shape=}")

        # Attention
        x = scaled_dot_product_attention(q, k, v, self.is_causal)
        # Try non-causal since it's Decode SDPA now
        # x = scaled_dot_product_attention(q, k, v, is_causal=False)

        # Rearrange back
        # x = rearrange_temporal(q, B, T, H, W, self.heads, self.head_dim, "BHWh_Td")
        x = x.reshape(B, H, W, self.heads, T, self.head_dim).transpose(0,4, 1, 2, 3, 5).reshape(B, T, H, W, self.heads * self.head_dim)
        # Project out
        x = self.to_out(x)
        return x#, (k, v)


class SpatialAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: object,  # Replace with MLX RotaryEmbedding when available
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)
        
        self.rotary_emb = rotary_emb

    def __call__(self, x):
        B, T, H, W, D = x.shape

        # Generate Q, K, V
        qkv = self.to_qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        # Rearrange for spatial attention
        q = q.reshape(B, T, H, W, self.heads, self.head_dim).transpose(0,1,4,2,3,5).reshape(B*T, self.heads, H, W, self.head_dim)
        k = k.reshape(B, T, H, W, self.heads, self.head_dim).transpose(0,1,4,2,3,5).reshape(B*T, self.heads, H, W, self.head_dim)
        v = v.reshape(B, T, H, W, self.heads, self.head_dim).transpose(0,1,4,2,3,5).reshape(B*T, self.heads, H, W, self.head_dim)

        # Apply rotary embeddings (implementation would depend on MLX version)
        if self.rotary_emb is not None:
            freqs = self.rotary_emb.get_axial_freqs(H, W)
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)

        # Reshape for attention
        q = q.reshape(B * T, self.heads, H * W, self.head_dim)
        k = k.reshape(B * T, self.heads, H * W, self.head_dim)
        v = v.reshape(B * T, self.heads, H * W, self.head_dim)

        # Attention
        x = scaled_dot_product_attention(q, k, v, is_causal=False)

        # Rearrange back
        x = x.reshape(B, T, self.heads, H, W, self.head_dim).transpose(0,1,3,4,2,5).reshape(B, T, H, W, self.heads * self.head_dim) 

        # Project out
        x = self.to_out(x)
        return x 