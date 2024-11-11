"""
MLX implementation of DiT (Diffusion Transformer)
"""

from typing import Optional
import mlx.core as mx
import mlx.nn as nn
from einops import rearrange
from patch_embed_mlx import PatchEmbed
from timestep_embedder_mlx import TimestepEmbedder
from rotary_embedding_mlx import RotaryEmbedding
from dit_block_mlx import SpatioTemporalDiTBlock, modulate, gate

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )

    def __call__(self, x, c):
        shift, scale = mx.split(self.adaLN_modulation(c), 2, axis=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_h=18,
        input_w=32,
        patch_size=2,
        in_channels=16,
        hidden_size=1024,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        external_cond_dim=25,
        max_frames=32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.max_frames = max_frames

        self.x_embedder = PatchEmbed(input_h, input_w, patch_size, in_channels, hidden_size, flatten=False)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.spatial_rotary_emb = RotaryEmbedding(
            dim=hidden_size // num_heads // 2, 
            freqs_for="pixel",
            max_freq=256
        )
        self.temporal_rotary_emb = RotaryEmbedding(dim=hidden_size // num_heads)
        
        if external_cond_dim > 0:
            self.external_cond = nn.Linear(external_cond_dim, hidden_size)
        else:
            self.external_cond = lambda x: x  # Identity function

        # Create transformer blocks
        self.blocks = [
            SpatioTemporalDiTBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                is_causal=True,
                spatial_rotary_emb=self.spatial_rotary_emb,
                temporal_rotary_emb=self.temporal_rotary_emb,
            )
            for _ in range(depth)
        ]

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

    def unpatchify(self, x):
        """
        x: (N, H, W, patch_size**2 * C)
        imgs: (N, C, H*P, W*P)
        """
        c = self.out_channels
        p = self.patch_size
        h = x.shape[1]
        w = x.shape[2]

        # Reshape to separate patch dimensions
        x = x.reshape(x.shape[0], h, w, p, p, c)
        
        # Equivalent to torch.einsum("nhwpqc->nchpwq", x)
        x = x.transpose(0, 5, 1, 3, 2, 4)
        
        # Merge patch dimensions
        imgs = x.reshape(x.shape[0], c, h * p, w * p)
        return imgs

    def __call__(self, x, t, external_cond=None, kv_cache=None):
        """
        Forward pass of DiT.
        x: (B, T, C, H, W) tensor of spatial inputs
        t: (B, T,) tensor of diffusion timesteps
        """
        B, T, C, H, W = x.shape
        # Add spatial embeddings
        # x = rearrange(x, "b t c h w -> (b t) c h w")
        x = x.reshape(B*T, C, H, W)
        x = self.x_embedder(x)
        _, H, W, D = x.shape
        x = x.reshape(-1, T, H, W, D)
        B, T, H, W, D = x.shape


        # Embed noise steps
        t = t.reshape(B*T)
        c = self.t_embedder(t)
        c = c.reshape(B, T, -1)

        if external_cond is not None:
            c = c + self.external_cond(external_cond)

        # Apply transformer blocks
        # if kv_cache is None:
        #     kv_cache = [None] * len(self.blocks)
        for idx, block in enumerate(self.blocks):
            x = block(x, c, kv_cache=None)

        # Final layer and unpatchify
        x = self.final_layer(x, c)
        B, T, H, W, D = x.shape
        x = x.reshape(B*T, H, W, D)
        x = self.unpatchify(x)
        BT, C, H, W = x.shape
        x = x.reshape(BT//T, T, C, H, W)

        return x
    
    def load_weights(self, torch_model):
        self.x_embedder.proj.weight = mx.array(torch_model.x_embedder.proj.weight.data.permute(0,2,3,1))
        self.x_embedder.proj.bias = mx.array(torch_model.x_embedder.proj.bias.data)

        # TimestepEmbedder weights
        self.t_embedder.mlp["layers"][0].weight = mx.array(torch_model.t_embedder.mlp[0].weight.data)
        self.t_embedder.mlp["layers"][0].bias = mx.array(torch_model.t_embedder.mlp[0].bias.data)
        self.t_embedder.mlp["layers"][2].weight = mx.array(torch_model.t_embedder.mlp[2].weight.data)
        self.t_embedder.mlp["layers"][2].bias = mx.array(torch_model.t_embedder.mlp[2].bias.data)

        # External condition weights
        self.external_cond.weight = mx.array(torch_model.external_cond.weight.data)
        self.external_cond.bias = mx.array(torch_model.external_cond.bias.data)

        # Copy weights for each transformer block
        for i in range(len(self.blocks)):
            # Spatial attention components
            self.blocks[i].s_attn.to_qkv.weight = mx.array(torch_model.blocks[i].s_attn.to_qkv.weight.data)
            self.blocks[i].s_attn.to_out.weight = mx.array(torch_model.blocks[i].s_attn.to_out.weight.data)
            self.blocks[i].s_attn.to_out.bias = mx.array(torch_model.blocks[i].s_attn.to_out.bias.data)
            
            # Spatial MLP components
            self.blocks[i].s_mlp.fc1.weight = mx.array(torch_model.blocks[i].s_mlp.fc1.weight.data)
            self.blocks[i].s_mlp.fc1.bias = mx.array(torch_model.blocks[i].s_mlp.fc1.bias.data)
            self.blocks[i].s_mlp.fc2.weight = mx.array(torch_model.blocks[i].s_mlp.fc2.weight.data)
            self.blocks[i].s_mlp.fc2.bias = mx.array(torch_model.blocks[i].s_mlp.fc2.bias.data)
            
            # Spatial AdaLN components
            self.blocks[i].s_adaLN_modulation["layers"][1].weight = mx.array(torch_model.blocks[i].s_adaLN_modulation[1].weight.data)
            self.blocks[i].s_adaLN_modulation["layers"][1].bias = mx.array(torch_model.blocks[i].s_adaLN_modulation[1].bias.data)

            # Temporal attention components
            self.blocks[i].t_attn.to_qkv.weight = mx.array(torch_model.blocks[i].t_attn.to_qkv.weight.data)
            self.blocks[i].t_attn.to_out.weight = mx.array(torch_model.blocks[i].t_attn.to_out.weight.data)
            self.blocks[i].t_attn.to_out.bias = mx.array(torch_model.blocks[i].t_attn.to_out.bias.data)
            
            # Temporal MLP components
            self.blocks[i].t_mlp.fc1.weight = mx.array(torch_model.blocks[i].t_mlp.fc1.weight.data)
            self.blocks[i].t_mlp.fc1.bias = mx.array(torch_model.blocks[i].t_mlp.fc1.bias.data)
            self.blocks[i].t_mlp.fc2.weight = mx.array(torch_model.blocks[i].t_mlp.fc2.weight.data)
            self.blocks[i].t_mlp.fc2.bias = mx.array(torch_model.blocks[i].t_mlp.fc2.bias.data)
            
            # Temporal AdaLN components
            self.blocks[i].t_adaLN_modulation["layers"][1].weight = mx.array(torch_model.blocks[i].t_adaLN_modulation[1].weight.data)
            self.blocks[i].t_adaLN_modulation["layers"][1].bias = mx.array(torch_model.blocks[i].t_adaLN_modulation[1].bias.data)

        # Final layer weights
        self.final_layer.linear.weight = mx.array(torch_model.final_layer.linear.weight.data)
        self.final_layer.linear.bias = mx.array(torch_model.final_layer.linear.bias.data)
        self.final_layer.adaLN_modulation["layers"][1].weight = mx.array(torch_model.final_layer.adaLN_modulation[1].weight.data)
        self.final_layer.adaLN_modulation["layers"][1].bias = mx.array(torch_model.final_layer.adaLN_modulation[1].bias.data)

def DiT_S_2():
    return DiT(
        patch_size=2,
        hidden_size=1024,
        depth=16,
        num_heads=16,
    )

DiT_models = {"DiT-S/2": DiT_S_2} 