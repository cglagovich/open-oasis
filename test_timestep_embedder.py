import pytest
import torch
import torch.nn as nn
import mlx.core as mx
import mlx.nn as mxnn
import numpy as np
from timestep_embedder_mlx import TimestepEmbedder as TimestepEmbedderMLX
from dit import TimestepEmbedder as TimestepEmbedderTorch

def assert_close(mlx_tensor, torch_tensor, atol=1e-6):
    """Compare MLX and PyTorch tensors for approximate equality"""
    mlx_np = mlx_tensor.tolist()
    torch_np = torch_tensor.detach().cpu().numpy()
    np.testing.assert_allclose(mlx_np, torch_np, atol=atol)

@pytest.mark.parametrize("hidden_size,frequency_embedding_size", [
    (768, 256),
    (1024, 512),
    (384, 128),
])
def test_timestep_embedder(hidden_size, frequency_embedding_size):
    # Initialize both implementations
    torch_model = TimestepEmbedderTorch(
        hidden_size=hidden_size,
        frequency_embedding_size=frequency_embedding_size
    )
    
    mlx_model = TimestepEmbedderMLX(
        hidden_size=hidden_size,
        frequency_embedding_size=frequency_embedding_size
    )

    # Copy weights from torch model to mlx model
    mlx_model.mlp["layers"][0].weight = mx.array(torch_model.mlp[0].weight.data.numpy())
    mlx_model.mlp["layers"][0].bias = mx.array(torch_model.mlp[0].bias.data.numpy())
    mlx_model.mlp["layers"][2].weight = mx.array(torch_model.mlp[2].weight.data.numpy())
    mlx_model.mlp["layers"][2].bias = mx.array(torch_model.mlp[2].bias.data.numpy())

    # Create input tensors
    batch_size = 4
    t_np = np.random.randint(0, 1000, size=(batch_size,)).astype(np.float32)
    t_torch = torch.from_numpy(t_np)
    t_mlx = mx.array(t_np)

    # Forward pass
    out_torch = torch_model(t_torch)
    out_mlx = mlx_model(t_mlx)

    # Check shapes match
    assert out_mlx.shape == tuple(out_torch.shape)

    # Check values are close
    assert_close(out_mlx, out_torch, atol=1e-4)

@pytest.mark.parametrize("batch_size,dim", [
    (1, 256),
    (4, 512),
    (8, 128),
])
def test_timestep_embedding(batch_size, dim):
    # Create identical random inputs
    t_np = np.random.randint(0, 1000, size=(batch_size,)).astype(np.float32)
    t_torch = torch.from_numpy(t_np)
    t_mlx = mx.array(t_np)
    
    # Apply both implementations
    out_torch = TimestepEmbedderTorch.timestep_embedding(t_torch, dim)
    out_mlx = TimestepEmbedderMLX.timestep_embedding(t_mlx, dim)
    
    # Compare results
    assert_close(out_mlx, out_torch, atol=1e-4)

if __name__ == "__main__":
    pytest.main([__file__]) 