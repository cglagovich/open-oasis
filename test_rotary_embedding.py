import pytest
import torch
import mlx.core as mx
import numpy as np
from rotary_embedding_mlx import rotate_half as rotate_half_mlx
from rotary_embedding_mlx import apply_rotary_emb as apply_rotary_emb_mlx
from rotary_embedding_mlx import RotaryEmbedding as RotaryEmbeddingMLX
from rotary_embedding_torch import rotate_half as rotate_half_torch
from rotary_embedding_torch import apply_rotary_emb as apply_rotary_emb_torch
from rotary_embedding_torch import RotaryEmbedding as RotaryEmbeddingTorch

def assert_close(mlx_tensor, torch_tensor, atol=1e-6):
    """Compare MLX and PyTorch tensors for approximate equality"""
    mlx_np = mlx_tensor.tolist()
    torch_np = torch_tensor.detach().cpu().numpy()
    np.testing.assert_allclose(mlx_np, torch_np, atol=atol)

@pytest.mark.parametrize("shape", [
    (2, 4),
    (2, 3, 4),
    (2, 4, 6),
])
def test_rotate_half(shape):
    # Create identical random inputs
    x_np = np.random.randn(*shape)
    x_torch = torch.from_numpy(x_np)
    x_mlx = mx.array(x_np)
    
    # Apply both implementations
    out_torch = rotate_half_torch(x_torch)
    out_mlx = rotate_half_mlx(x_mlx)
    
    # Compare results
    assert_close(out_mlx, out_torch)

@pytest.mark.parametrize("shape,scale,seq_dim", [
    ((2, 4, 6), 1.0, -2),
    ((2, 3, 8), 0.5, -2),
    ((2, 4, 6), 1.0, -1),
])
def test_apply_rotary_emb(shape, scale, seq_dim):
    # Create identical random inputs
    t_np = np.random.randn(*shape)
    freqs_np = np.random.randn(*shape)
    
    t_torch = torch.from_numpy(t_np)
    freqs_torch = torch.from_numpy(freqs_np)
    
    t_mlx = mx.array(t_np)
    freqs_mlx = mx.array(freqs_np)
    
    # Apply both implementations
    out_torch = apply_rotary_emb_torch(freqs_torch, t_torch, scale=scale, seq_dim=seq_dim)
    out_mlx = apply_rotary_emb_mlx(freqs_mlx, t_mlx, scale=scale, seq_dim=seq_dim)
    
    # Compare results
    assert_close(out_mlx, out_torch)

@pytest.mark.parametrize("dim,freqs_for,max_freq", [
    (64, "pixel", 256), # spatial_rotary_emb
    (64, "lang", None), #temporal_rotary_emb
    (64//4, "pixel", 18*32), # vae
])
@pytest.mark.parametrize("seq_len, freqs_len", [
    (18, 8),
    (32, 8),
    (9, 16),
    (16, 2),
    (2, 32),
    (2, 16),
    (16, 16),
])
def test_rotary_embedding_forward(dim, freqs_for, max_freq, seq_len, freqs_len):
    # Initialize both implementations
    rope_torch = RotaryEmbeddingTorch(
        dim=dim,
        freqs_for=freqs_for,
        max_freq=max_freq,
    )
    
    rope_mlx = RotaryEmbeddingMLX(
        dim=dim,
        freqs_for=freqs_for,
        max_freq=max_freq,
    )
    
    # Test sequence positions
    t = torch.randn(seq_len)
    freqs = torch.randn(freqs_len)

    t_mlx = mx.array(t)
    freqs_mlx = mx.array(freqs)
    
    # Get frequencies from both implementations
    freqs_torch = rope_torch.forward(t, freqs)
    freqs_mlx = rope_mlx.forward(t_mlx, freqs_mlx)
    
    # Compare results
    assert_close(freqs_mlx, freqs_torch)

@pytest.mark.parametrize("dim,freqs_for,max_freq", [
    (64, "pixel", 256), # spatial_rotary_emb
    (64, "lang", None), #temporal_rotary_emb
])
@pytest.mark.parametrize("t_shape", [
    ((144, 16, 2, 64)),
])
def test_rotary_embedding_rotate(dim, freqs_for, max_freq, t_shape):
    # Initialize both implementations
    rope_torch = RotaryEmbeddingTorch(
        dim=dim,
        freqs_for=freqs_for,
        max_freq=max_freq,
    )
    
    rope_mlx = RotaryEmbeddingMLX(
        dim=dim,
        freqs_for=freqs_for,
        max_freq=max_freq,
    )

    assert_close(rope_mlx.freqs, rope_torch.freqs, atol=1e-5)
    
    # Test rotation of queries/keys
    # Create tensor with shape matching file_context_0
    q_np = np.random.randn(*t_shape)
    q_torch = torch.from_numpy(q_np)
    q_mlx = mx.array(q_np)
    
    # Rotate queries with parameters from file_context_0
    rotated_q_torch = rope_torch.rotate_queries_or_keys(
        q_torch, 
        rope_torch.freqs,
    )
    rotated_q_mlx = rope_mlx.rotate_queries_or_keys(
        q_mlx,
        rope_mlx.freqs,
    )
    
    # Compare results
    assert_close(rotated_q_mlx, rotated_q_torch, atol=1e-3)

@pytest.mark.parametrize("dim,freqs_for,max_freq", [
    (64, "pixel", 256), # spatial_rotary_emb
])
@pytest.mark.parametrize("H,W", [
    ((8, 16)),
])
def test_rotary_embedding_axial_freqs(dim, freqs_for, max_freq, H, W):
    # Initialize both implementations
    rope_torch = RotaryEmbeddingTorch(
        dim=dim,
        freqs_for=freqs_for,
        max_freq=max_freq,
    )
    
    rope_mlx = RotaryEmbeddingMLX(
        dim=dim,
        freqs_for=freqs_for,
        max_freq=max_freq,
    )

    assert_close(rope_mlx.freqs, rope_torch.freqs, atol=1e-5)
    
    torch_axial_freqs = rope_torch.get_axial_freqs(H, W)
    mlx_axial_freqs = rope_mlx.get_axial_freqs(H, W)
    
    # Compare results
    assert_close(mlx_axial_freqs, torch_axial_freqs, atol=1e-4)

if __name__ == "__main__":
    pytest.main([__file__]) 