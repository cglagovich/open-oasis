import pytest
import torch
import torch.nn as nn
import mlx.core as mx
import mlx.nn as mxnn
import numpy as np
from einops import rearrange
from patch_embed_mlx import PatchEmbed as PatchEmbedMLX
from dit import PatchEmbed as PatchEmbedTorch

def assert_close(mlx_tensor, torch_tensor, atol=1e-6):
    """Compare MLX and PyTorch tensors for approximate equality"""
    mlx_np = mlx_tensor.tolist()
    torch_np = torch_tensor.detach().cpu().numpy()
    np.testing.assert_allclose(mlx_np, torch_np, atol=atol)

@pytest.mark.parametrize("img_size,patch_size,in_chans,embed_dim,flatten", [
    ((256, 256), 16, 3, 768, True),
    ((224, 224), 32, 1, 384, False),
    ((384, 384), 8, 3, 512, True),
])
def test_patch_embed(img_size, patch_size, in_chans, embed_dim, flatten):
    # Initialize both implementations
    torch_model = PatchEmbedTorch(
        img_height=img_size[0],
        img_width=img_size[1],
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        norm_layer=nn.LayerNorm,
        flatten=flatten
    )
    
    mlx_model = PatchEmbedMLX(
        img_height=img_size[0],
        img_width=img_size[1],
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        norm_layer=mxnn.LayerNorm,
        flatten=flatten
    )

    # # Copy weights from torch model to mlx model
    # Transform torch weights to match shape expected by mlx model
    print(torch_model.proj.weight.shape)
    print(mlx_model.proj.weight.shape)
    mlx_model.proj.weight = mx.array(torch_model.proj.weight.data.permute(0,2,3,1).numpy())
    mlx_model.proj.bias = mx.array(torch_model.proj.bias.data.numpy())
    mlx_model.norm.weight = mx.array(torch_model.norm.weight.data.numpy())
    mlx_model.norm.bias = mx.array(torch_model.norm.bias.data.numpy())

    # Create input tensors
    batch_size = 2
    x_np = np.random.randn(batch_size, in_chans, img_size[0], img_size[1]).astype(np.float32)
    x_torch = torch.from_numpy(x_np)
    x_mlx = mx.array(x_np)

    # Forward pass
    out_torch = torch_model(x_torch)
    out_mlx = mlx_model(x_mlx)

    # Check shapes match
    assert out_mlx.shape == tuple(out_torch.shape)

    # Check values are close
    assert_close(out_mlx, out_torch, atol=1e-5)

# def test_random_sample():
#     """Test that random_sample=True allows different input sizes"""
#     img_size = (256, 256)
#     patch_size = 16
#     in_chans = 3
#     embed_dim = 768
    
#     torch_model = PatchEmbedTorch(
#         img_height=img_size[0],
#         img_width=img_size[1],
#         patch_size=patch_size,
#         in_chans=in_chans,
#         embed_dim=embed_dim
#     )
    
#     mlx_model = PatchEmbedMLX(
#         img_height=img_size[0],
#         img_width=img_size[1],
#         patch_size=patch_size,
#         in_chans=in_chans,
#         embed_dim=embed_dim
#     )

#     # Copy weights
#     mlx_model.proj.weight = mx.array(torch_model.proj.weight.data.numpy())
#     mlx_model.proj.bias = mx.array(torch_model.proj.bias.data.numpy())
#     mlx_model.norm.weight = mx.array(torch_model.norm.weight.data.numpy())
#     mlx_model.norm.bias = mx.array(torch_model.norm.bias.data.numpy())

#     # Test with different input size
#     different_size = (224, 224)
#     batch_size = 2
#     x_np = np.random.randn(batch_size, in_chans, different_size[0], different_size[1]).astype(np.float32)
#     x_torch = torch.from_numpy(x_np)
#     x_mlx = mx.array(x_np)

#     # Should work with random_sample=True
#     out_torch = torch_model(x_torch, random_sample=True)
#     out_mlx = mlx_model(x_mlx, random_sample=True)

#     assert_close(out_mlx, out_torch, atol=1e-5)

#     # Should fail with random_sample=False
#     with pytest.raises(AssertionError):
#         torch_model(x_torch, random_sample=False)
#     with pytest.raises(AssertionError):
#         mlx_model(x_mlx, random_sample=False) 