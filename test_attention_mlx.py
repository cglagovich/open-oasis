import pytest
import numpy as np
import torch
import mlx.core as mx
import mlx.nn as nn

from attention_mlx import TemporalAxialAttention, SpatialAxialAttention
from rotary_embedding_mlx import RotaryEmbedding as RotaryEmbedding_MLX
from attention import TemporalAxialAttention as TorchTemporalAxialAttention
from attention import SpatialAxialAttention as TorchSpatialAxialAttention
from rotary_embedding_torch import RotaryEmbedding as RotaryEmbedding_Torch
@pytest.fixture
def test_config():
    return {
        'batch_size': 2,
        'time_steps': 4,
        'height': 8,
        'width': 16,
        'dim': 64,
        'heads': 4,
        'dim_head': 16,
    }

def test_temporal_attention_shapes(test_config):
    B, T, H, W = (test_config['batch_size'], test_config['time_steps'],
                  test_config['height'], test_config['width'])
    dim = test_config['dim']
    
    # Initialize both implementations
    torch_rope = RotaryEmbedding_Torch(dim=dim // test_config['heads'])
    torch_model = TorchTemporalAxialAttention(
        dim=dim,
        heads=test_config['heads'],
        dim_head=test_config['dim_head'],
        rotary_emb=torch_rope,
    )
    mlx_rope = RotaryEmbedding_MLX(dim=dim // test_config['heads'])
    mlx_model = TemporalAxialAttention(
        dim=dim,
        heads=test_config['heads'],
        dim_head=test_config['dim_head'],
        rotary_emb=mlx_rope,
    )

    # Load weights from torch model to mlx model. Include bias. Convert torch parameters to mlx parameters.
    mlx_model.to_qkv.weight = mx.array(torch_model.to_qkv.weight.data)
    mlx_model.to_out.weight = mx.array(torch_model.to_out.weight.data)
    mlx_model.to_out.bias = mx.array(torch_model.to_out.bias.data)

    
    # Create input tensors
    x_np = np.random.randn(B, T, H, W, dim).astype(np.float32)
    x_mlx = mx.array(x_np)
    x_torch = torch.from_numpy(x_np)
    
    # Forward pass
    out_mlx = mlx_model(x_mlx)
    out_torch = torch_model(x_torch)
    
    # Convert outputs to numpy for comparison
    out_torch_np = out_torch.detach().numpy()
    
    # Check shapes
    assert out_mlx.shape == out_torch_np.shape
    # assert out_mlx.shape == (B, T, H, W, dim)
    
    # Check values are close (allowing for small numerical differences)
    np.testing.assert_allclose(out_mlx, out_torch_np, rtol=1e-4, atol=1e-4)

def test_spatial_attention_shapes(test_config):
    B, T, H, W = (test_config['batch_size'], test_config['time_steps'],
                  test_config['height'], test_config['width'])
    dim = test_config['dim']
    
    # Initialize both implementations
    mlx_rope = RotaryEmbedding_MLX(dim=dim // test_config['heads']//2, freqs_for="pixel", max_freq=256)
    mlx_model = SpatialAxialAttention(
        dim=dim,
        heads=test_config['heads'],
        dim_head=test_config['dim_head'],
        rotary_emb=mlx_rope,
    )

    torch_rope = RotaryEmbedding_Torch(dim=dim // test_config['heads']//2, freqs_for="pixel", max_freq=256)
    torch_model = TorchSpatialAxialAttention(
        dim=dim,
        heads=test_config['heads'],
        dim_head=test_config['dim_head'],
        rotary_emb=torch_rope,
    )

    # Load weights from torch model to mlx model. Include bias. Convert torch parameters to mlx parameters.
    mlx_model.to_qkv.weight = mx.array(torch_model.to_qkv.weight.data)
    mlx_model.to_out.weight = mx.array(torch_model.to_out.weight.data)
    mlx_model.to_out.bias = mx.array(torch_model.to_out.bias.data)
    
    # Create input tensors
    x_np = np.random.randn(B, T, H, W, dim).astype(np.float32)
    x_mlx = mx.array(x_np)
    x_torch = torch.from_numpy(x_np)
    
    # Forward pass
    out_mlx = mlx_model(x_mlx)
    out_torch = torch_model(x_torch)
    
    # Convert outputs to numpy for comparison
    out_torch_np = out_torch.detach().numpy()
    
    # Check shapes
    assert out_mlx.shape == out_torch_np.shape
    assert out_mlx.shape == (B, T, H, W, dim)
    
    # Check values are close (allowing for small numerical differences)
    np.testing.assert_allclose(out_mlx, out_torch_np, rtol=1e-4, atol=1e-5)
