import mlx.core as mx
import mlx.nn as nn

def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return x
    return (x, x)

class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""
    
    def __init__(
        self,
        img_height=256,
        img_width=256,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        img_size = (img_height, img_width)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # MLX uses NCHW format for convolutions like PyTorch
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def __call__(self, x, random_sample=False):
        # MLX doesn't have shape attributes, so we get dimensions directly
        assert not random_sample, "Random sampling not supported"
        B, C, H, W = x.shape
        
        if not random_sample:
            assert H == self.img_size[0] and W == self.img_size[1], \
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # Reshape to NHWC (channels last)
        x = x.transpose(0, 2, 3, 1)
        x = self.proj(x)
        
        # MLX doesn't have rearrange built-in, but we can use einops
        B, H, W, C = x.shape
        if self.flatten:
            # x = rearrange(x, "B C H W -> B (H W) C")
            x = x.reshape(B, -1, C)
            
        x = self.norm(x)
        return x 