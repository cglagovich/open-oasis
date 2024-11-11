import pytest
import torch
import mlx.core as mx
import mlx.nn as mxnn
import numpy as np
from einops import rearrange

from dit_mlx import DiT as MLXDiT
from dit import DiT as TorchDiT
from rotary_embedding_mlx import RotaryEmbedding as MLXRotaryEmbedding
from rotary_embedding_torch import RotaryEmbedding as TorchRotaryEmbedding

def assert_close(mlx_tensor, torch_tensor, atol=1e-6):
    """Compare MLX and PyTorch tensors for approximate equality"""
    mlx_np = mlx_tensor.tolist()
    torch_np = torch_tensor.detach().cpu().numpy()
    np.testing.assert_allclose(mlx_np, torch_np, atol=atol)

@pytest.mark.parametrize("hidden_size,num_heads,depth", [
    (1024, 16, 2),
])
def test_dit(hidden_size, num_heads, depth):
    # Test configuration
    B, T, C, H, W = 1, 2, 16, 18, 32
    patch_size = 2
    mlp_ratio = 4.0
    external_cond_dim = 25

    # Initialize both implementations
    torch_model = TorchDiT(
        input_h=H,
        input_w=W,
        patch_size=patch_size,
        in_channels=C,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        external_cond_dim=external_cond_dim,
    )
    
    mlx_model = MLXDiT(
        input_h=H,
        input_w=W,
        patch_size=patch_size,
        in_channels=C,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        external_cond_dim=external_cond_dim,
    )

    # Copy weights from torch model to mlx model
    # PatchEmbed weights
    mlx_model.x_embedder.proj.weight = mx.array(torch_model.x_embedder.proj.weight.data.permute(0,2,3,1))
    mlx_model.x_embedder.proj.bias = mx.array(torch_model.x_embedder.proj.bias.data)

    # TimestepEmbedder weights
    mlx_model.t_embedder.mlp["layers"][0].weight = mx.array(torch_model.t_embedder.mlp[0].weight.data)
    mlx_model.t_embedder.mlp["layers"][0].bias = mx.array(torch_model.t_embedder.mlp[0].bias.data)
    mlx_model.t_embedder.mlp["layers"][2].weight = mx.array(torch_model.t_embedder.mlp[2].weight.data)
    mlx_model.t_embedder.mlp["layers"][2].bias = mx.array(torch_model.t_embedder.mlp[2].bias.data)

    # External condition weights
    if external_cond_dim > 0:
        mlx_model.external_cond.weight = mx.array(torch_model.external_cond.weight.data)
        mlx_model.external_cond.bias = mx.array(torch_model.external_cond.bias.data)

    # Copy weights for each transformer block
    for i in range(depth):
        # Spatial attention components
        mlx_model.blocks[i].s_attn.to_qkv.weight = mx.array(torch_model.blocks[i].s_attn.to_qkv.weight.data)
        mlx_model.blocks[i].s_attn.to_out.weight = mx.array(torch_model.blocks[i].s_attn.to_out.weight.data)
        mlx_model.blocks[i].s_attn.to_out.bias = mx.array(torch_model.blocks[i].s_attn.to_out.bias.data)
        
        # Spatial MLP components
        mlx_model.blocks[i].s_mlp.fc1.weight = mx.array(torch_model.blocks[i].s_mlp.fc1.weight.data)
        mlx_model.blocks[i].s_mlp.fc1.bias = mx.array(torch_model.blocks[i].s_mlp.fc1.bias.data)
        mlx_model.blocks[i].s_mlp.fc2.weight = mx.array(torch_model.blocks[i].s_mlp.fc2.weight.data)
        mlx_model.blocks[i].s_mlp.fc2.bias = mx.array(torch_model.blocks[i].s_mlp.fc2.bias.data)
        
        # Spatial AdaLN components
        mlx_model.blocks[i].s_adaLN_modulation["layers"][1].weight = mx.array(torch_model.blocks[i].s_adaLN_modulation[1].weight.data)
        mlx_model.blocks[i].s_adaLN_modulation["layers"][1].bias = mx.array(torch_model.blocks[i].s_adaLN_modulation[1].bias.data)

        # Temporal attention components
        mlx_model.blocks[i].t_attn.to_qkv.weight = mx.array(torch_model.blocks[i].t_attn.to_qkv.weight.data)
        mlx_model.blocks[i].t_attn.to_out.weight = mx.array(torch_model.blocks[i].t_attn.to_out.weight.data)
        mlx_model.blocks[i].t_attn.to_out.bias = mx.array(torch_model.blocks[i].t_attn.to_out.bias.data)
        
        # Temporal MLP components
        mlx_model.blocks[i].t_mlp.fc1.weight = mx.array(torch_model.blocks[i].t_mlp.fc1.weight.data)
        mlx_model.blocks[i].t_mlp.fc1.bias = mx.array(torch_model.blocks[i].t_mlp.fc1.bias.data)
        mlx_model.blocks[i].t_mlp.fc2.weight = mx.array(torch_model.blocks[i].t_mlp.fc2.weight.data)
        mlx_model.blocks[i].t_mlp.fc2.bias = mx.array(torch_model.blocks[i].t_mlp.fc2.bias.data)
        
        # Temporal AdaLN components
        mlx_model.blocks[i].t_adaLN_modulation["layers"][1].weight = mx.array(torch_model.blocks[i].t_adaLN_modulation[1].weight.data)
        mlx_model.blocks[i].t_adaLN_modulation["layers"][1].bias = mx.array(torch_model.blocks[i].t_adaLN_modulation[1].bias.data)

    # Final layer weights
    mlx_model.final_layer.linear.weight = mx.array(torch_model.final_layer.linear.weight.data)
    mlx_model.final_layer.linear.bias = mx.array(torch_model.final_layer.linear.bias.data)
    mlx_model.final_layer.adaLN_modulation["layers"][1].weight = mx.array(torch_model.final_layer.adaLN_modulation[1].weight.data)
    mlx_model.final_layer.adaLN_modulation["layers"][1].bias = mx.array(torch_model.final_layer.adaLN_modulation[1].bias.data)

    # Create input tensors
    x_np = np.random.randn(B, T, C, H, W).astype(np.float32)
    t_np = np.random.randint(0, 1000, size=(B, T)).astype(np.float32)
    external_cond_np = np.random.randn(B, T, external_cond_dim).astype(np.float32)
    
    x_torch = torch.from_numpy(x_np)
    t_torch = torch.from_numpy(t_np)
    external_cond_torch = torch.from_numpy(external_cond_np)
    
    x_mlx = mx.array(x_np)
    t_mlx = mx.array(t_np)
    external_cond_mlx = mx.array(external_cond_np)

    # Forward pass
    with torch.no_grad():
        out_torch = torch_model(x_torch, t_torch, external_cond_torch)
    out_mlx = mlx_model(x_mlx, t_mlx, external_cond_mlx)

    # Check shapes match
    assert out_mlx.shape == tuple(out_torch.shape)
    
    # Check values are close
    assert_close(out_mlx, out_torch, atol=1e-4)

if __name__ == "__main__":
    pytest.main([__file__]) 