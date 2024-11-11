"""
MLX implementation of rotary embeddings, adapted from the PyTorch version
"""

from typing import Literal, Optional
from math import pi
import mlx.core as mx
import mlx.nn as nn

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def rotate_half(x):
    # Reshape to separate the rotation dimension
    shape = x.shape
    d = shape[-1] // 2
    x = x.reshape(*shape[:-1], d, 2)
    
    # Split and stack
    x1, x2 = x[..., 0], x[..., 1]
    rotated = mx.stack([-x2, x1], axis=-1)
    
    # Restore original shape
    return rotated.reshape(*shape)

def apply_rotary_emb(freqs, t, scale=1.0, seq_dim=-2):
    start_index = 0
    if len(t.shape) == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[-1], f"feature dimension {t.shape[-1]} is not sufficient for rotation positions {rot_dim}"

    # Split tensor into parts
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]

    # Apply rotary embeddings
    t_transformed = (t_middle * mx.cos(freqs) * scale) + (rotate_half(t_middle) * mx.sin(freqs) * scale)

    # Concatenate back together
    return mx.concatenate([t_left, t_transformed, t_right], axis=-1)

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        freqs_for: Literal["lang", "pixel", "constant"] = "lang",
        theta: float = 10000,
        max_freq: float = 10,
        num_freqs: int = 1,
        interpolate_factor: float = 1.0,
        theta_rescale_factor: float = 1.0,
        seq_before_head_dim: bool = False,
    ):
        super().__init__()
        
        # Adjust theta based on rescale factor
        theta *= theta_rescale_factor ** (dim / (dim - 2))
        
        self.freqs_for = freqs_for
        self.dim = dim
        self.interpolate_factor = interpolate_factor
        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # Initialize frequencies
        if freqs_for == "lang":
            freqs = 1.0 / (theta ** (mx.arange(0, dim, 2)[:(dim // 2)] / dim))
        elif freqs_for == "pixel":
            freqs = mx.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = mx.ones((num_freqs,))
        
        self.freqs = mx.array(freqs)

    def get_seq_pos(self, seq_len, offset=0):
        return (mx.arange(seq_len) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, freqs, seq_dim=None, offset=0):
        seq_dim = default(seq_dim, self.default_seq_dim)
        seq_len = t.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, offset=offset)
        freqs = self.forward(seq, seq_len=seq_len, offset=offset)

        if seq_dim == -3:
            freqs = freqs.reshape(freqs.shape[0], 1, freqs.shape[1])

        return apply_rotary_emb(freqs, t, seq_dim=seq_dim)

    def forward(self, t, freqs=None, seq_len=None, offset=0):
        # MLX version doesn't need caching since computation is fast
        # Use self.freqs if freqs not provided
        freqs = default(freqs, self.freqs)
        
        # Handle arbitrary input dimensions by expanding and broadcasting
        # Equivalent operation without einsum
        freqs = mx.expand_dims(t, -1) * mx.expand_dims(freqs, 0)
        
        # freqs = freqs.reshape([1] * (t.ndim) + [-1])  # Add dims to match t
        # freqs = mx.broadcast_to(freqs, t.shape + (freqs.shape[-1],))  # Broadcast to final shape
        
        # Duplicate frequencies
        freqs = mx.repeat(freqs, 2, axis=-1)
        
        return freqs

    def get_axial_freqs(self, *dims):
        # Used by attention.py
        all_freqs = []

        for ind, dim in enumerate(dims):
            # only allow pixel freqs for last two dimensions
            use_pixel = (self.freqs_for == "pixel" or self.freqs_for == "spacetime") and ind >= len(dims) - 2
            if use_pixel:
                pos = mx.linspace(-1, 1, dim)
            else:
                pos = mx.arange(dim)

            if self.freqs_for == "spacetime" and not use_pixel:
                seq_freqs = self.forward(pos, self.time_freqs, seq_len=dim)
            else:
                seq_freqs = self.forward(pos, self.freqs, seq_len=dim)

            # Reshape seq_freqs to have the correct number of dimensions before broadcasting
            reshape_dims = [1] * len(dims)
            reshape_dims[ind] = dim
            seq_freqs = seq_freqs.reshape(*reshape_dims, -1)

            # Broadcast to full dimension size before appending
            expand_shape = [dims[i] if i != ind else dim for i in range(len(dims))]
            expand_shape.append(seq_freqs.shape[-1])
            seq_freqs = mx.broadcast_to(seq_freqs, expand_shape)
            all_freqs.append(seq_freqs)

        # Now arrays should have compatible shapes for concatenation
        return mx.concatenate(all_freqs, axis=-1)