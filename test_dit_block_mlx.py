import pytest
import torch
import mlx.core as mx
import mlx.nn as mxnn
import numpy as np
from einops import rearrange

from dit_block_mlx import SpatioTemporalDiTBlock as MLXDiTBlock
from dit import SpatioTemporalDiTBlock as TorchDiTBlock
from rotary_embedding_mlx import RotaryEmbedding as MLXRotaryEmbedding
from rotary_embedding_torch import RotaryEmbedding as TorchRotaryEmbedding

def assert_close(mlx_tensor, torch_tensor, atol=1e-6):
    """Compare MLX and PyTorch tensors for approximate equality"""
    mlx_np = mlx_tensor.tolist()
    torch_np = torch_tensor.detach().cpu().numpy()
    np.testing.assert_allclose(mlx_np, torch_np, atol=atol)

@pytest.mark.parametrize("hidden_size,num_heads,mlp_ratio", [
    (64, 4, 4.0),
    (128, 8, 4.0),
])
def test_dit_block(hidden_size, num_heads, mlp_ratio):
    # Test configuration
    batch_size = 2
    time_steps = 4
    height = 8
    width = 16
    
    # Initialize rotary embeddings
    spatial_rotary_emb_torch = TorchRotaryEmbedding(
        dim=hidden_size // num_heads // 2,
        freqs_for="pixel",
        max_freq=256
    )
    temporal_rotary_emb_torch = TorchRotaryEmbedding(
        dim=hidden_size // num_heads
    )
    
    spatial_rotary_emb_mlx = MLXRotaryEmbedding(
        dim=hidden_size // num_heads // 2,
        freqs_for="pixel",
        max_freq=256
    )
    temporal_rotary_emb_mlx = MLXRotaryEmbedding(
        dim=hidden_size // num_heads
    )

    # Initialize both implementations
    torch_model = TorchDiTBlock(
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        spatial_rotary_emb=spatial_rotary_emb_torch,
        temporal_rotary_emb=temporal_rotary_emb_torch,
    )
    
    mlx_model = MLXDiTBlock(
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        spatial_rotary_emb=spatial_rotary_emb_mlx,
        temporal_rotary_emb=temporal_rotary_emb_mlx,
    )

    # Copy weights from torch model to mlx model
    # Spatial attention components
    mlx_model.s_attn.to_qkv.weight = mx.array(torch_model.s_attn.to_qkv.weight.data)
    mlx_model.s_attn.to_out.weight = mx.array(torch_model.s_attn.to_out.weight.data)
    mlx_model.s_attn.to_out.bias = mx.array(torch_model.s_attn.to_out.bias.data)
    
    # Spatial MLP components
    mlx_model.s_mlp.fc1.weight = mx.array(torch_model.s_mlp.fc1.weight.data)
    mlx_model.s_mlp.fc1.bias = mx.array(torch_model.s_mlp.fc1.bias.data)
    mlx_model.s_mlp.fc2.weight = mx.array(torch_model.s_mlp.fc2.weight.data)
    mlx_model.s_mlp.fc2.bias = mx.array(torch_model.s_mlp.fc2.bias.data)
    
    # Spatial AdaLN components
    mlx_model.s_adaLN_modulation["layers"][1].weight = mx.array(torch_model.s_adaLN_modulation[1].weight.data)
    mlx_model.s_adaLN_modulation["layers"][1].bias = mx.array(torch_model.s_adaLN_modulation[1].bias.data)

    # Temporal attention components
    mlx_model.t_attn.to_qkv.weight = mx.array(torch_model.t_attn.to_qkv.weight.data)
    mlx_model.t_attn.to_out.weight = mx.array(torch_model.t_attn.to_out.weight.data)
    mlx_model.t_attn.to_out.bias = mx.array(torch_model.t_attn.to_out.bias.data)
    
    # Temporal MLP components
    mlx_model.t_mlp.fc1.weight = mx.array(torch_model.t_mlp.fc1.weight.data)
    mlx_model.t_mlp.fc1.bias = mx.array(torch_model.t_mlp.fc1.bias.data)
    mlx_model.t_mlp.fc2.weight = mx.array(torch_model.t_mlp.fc2.weight.data)
    mlx_model.t_mlp.fc2.bias = mx.array(torch_model.t_mlp.fc2.bias.data)
    
    # Temporal AdaLN components
    mlx_model.t_adaLN_modulation["layers"][1].weight = mx.array(torch_model.t_adaLN_modulation[1].weight.data)
    mlx_model.t_adaLN_modulation["layers"][1].bias = mx.array(torch_model.t_adaLN_modulation[1].bias.data)

    # Create input tensors
    x_np = np.random.randn(batch_size, time_steps, height, width, hidden_size).astype(np.float32)
    c_np = np.random.randn(batch_size, time_steps, hidden_size).astype(np.float32)
    
    x_torch = torch.from_numpy(x_np)
    c_torch = torch.from_numpy(c_np)
    
    x_mlx = mx.array(x_np)
    c_mlx = mx.array(c_np)

    # Forward pass
    with torch.no_grad():
        out_torch = torch_model(x_torch, c_torch)
    out_mlx = mlx_model(x_mlx, c_mlx)

    # Check shapes match
    assert out_mlx.shape == tuple(out_torch.shape)
    
    # Check values are close
    assert_close(out_mlx, out_torch, atol=1e-4)

if __name__ == "__main__":
    pytest.main([__file__]) 