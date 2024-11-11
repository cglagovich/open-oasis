"""
MLX implementation of SpatioTemporalDiTBlock
"""

from typing import Optional
import mlx.core as mx
import mlx.nn as nn
from mlp_mlx import Mlp
from attention_mlx import SpatialAxialAttention, TemporalAxialAttention
from rotary_embedding_mlx import RotaryEmbedding

def modulate(x, shift, scale):   
    """Apply modulation to input tensor"""
    # Handle broadcasting for shift and scale
    fixed_dims = [1] * (len(shift.shape) - 1)
    repeat_factor = x.shape[0] // shift.shape[0]
    shift = mx.repeat(shift, repeat_factor, axis=0)
    scale = mx.repeat(scale, repeat_factor, axis=0)
    
    # Add dimensions to match x
    while len(shift.shape) < len(x.shape):
        shift = mx.expand_dims(shift, -2)
        scale = mx.expand_dims(scale, -2)
    
    return x * (1 + scale) + shift

def gate(x, g):
    """Apply gating to input tensor"""
    fixed_dims = [1] * (len(g.shape) - 1)
    repeat_factor = x.shape[0] // g.shape[0]
    g = mx.repeat(g, repeat_factor, axis=0)
    
    while len(g.shape) < len(x.shape):
        g = mx.expand_dims(g, -2)
    
    return g * x

class SpatioTemporalDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        is_causal: bool = True,
        spatial_rotary_emb: Optional[RotaryEmbedding] = None,
        temporal_rotary_emb: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        self.is_causal = is_causal
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approx="tanh")

        # Spatial attention components
        self.s_norm1 = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        self.s_attn = SpatialAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            rotary_emb=spatial_rotary_emb,
        )
        self.s_norm2 = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        self.s_mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.s_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        # Temporal attention components
        self.t_norm1 = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        self.t_attn = TemporalAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            is_causal=is_causal,
            rotary_emb=temporal_rotary_emb,
        )
        self.t_norm2 = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        self.t_mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.t_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def __call__(self, x, c):
        B, T, H, W, D = x.shape

        # Spatial block
        s_modulation = self.s_adaLN_modulation(c)
        s_shift_msa, s_scale_msa, s_gate_msa, s_shift_mlp, s_scale_mlp, s_gate_mlp = mx.split(s_modulation, 6, axis=-1)
        
        # Spatial attention path
        x_norm1 = self.s_norm1(x)
        x_mod1 = modulate(x_norm1, s_shift_msa, s_scale_msa)
        x_attn = self.s_attn(x_mod1)
        x_gated = gate(x_attn, s_gate_msa)
        x = x + x_gated

        # Spatial MLP path
        x_norm2 = self.s_norm2(x)
        x_mod2 = modulate(x_norm2, s_shift_mlp, s_scale_mlp)
        x_mlp = self.s_mlp(x_mod2)
        x_gated = gate(x_mlp, s_gate_mlp)
        x = x + x_gated

        # Temporal block
        t_modulation = self.t_adaLN_modulation(c)
        t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp = mx.split(t_modulation, 6, axis=-1)
        
        # Temporal attention path
        x_norm1 = self.t_norm1(x)
        x_mod1 = modulate(x_norm1, t_shift_msa, t_scale_msa)
        x_attn = self.t_attn(x_mod1)
        x_gated = gate(x_attn, t_gate_msa)
        x = x + x_gated

        # Temporal MLP path
        x_norm2 = self.t_norm2(x)
        x_mod2 = modulate(x_norm2, t_shift_mlp, t_scale_mlp)
        x_mlp = self.t_mlp(x_mod2)
        x_gated = gate(x_mlp, t_gate_mlp)
        x = x + x_gated

        return x