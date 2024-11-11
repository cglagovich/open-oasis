import torch
from torch import nn
import pytest

from mlp_mlx import Mlp as Mlp_MLX
from timm.models.vision_transformer import Mlp
from test_rotary_embedding import assert_close

# import mlx nn
import mlx.core as mx
import mlx.nn as mxnn

@pytest.mark.parametrize("hidden_size, hidden_dim", [
    (1024, 4096),
])
def test_mlp(hidden_size, hidden_dim):
    s = 32
    torch_approx_gelu = lambda: nn.GELU(approximate="tanh")
    mlp_torch = Mlp(hidden_size, hidden_dim, act_layer=torch_approx_gelu, drop=0)
    mlx_approx_gelu = lambda: mxnn.GELU(approx="tanh")
    mlp_mlx = Mlp_MLX(hidden_size, hidden_dim, act_layer=mlx_approx_gelu, drop=0)
    # Load weights from torch model to mlx model. Include bias. Convert torch parameters to mlx parameters.
    mlp_mlx.fc1.weight = mx.array(mlp_torch.fc1.weight.data)
    mlp_mlx.fc1.bias = mx.array(mlp_torch.fc1.bias.data)
    mlp_mlx.fc2.weight = mx.array(mlp_torch.fc2.weight.data)
    mlp_mlx.fc2.bias = mx.array(mlp_torch.fc2.bias.data)
    x = torch.randn(s, hidden_size)
    x_mlx = mx.array(x)
    y_torch = mlp_torch(x)
    y_mlx = mlp_mlx(x_mlx)
    assert_close(y_mlx, y_torch, atol=1e-5)
