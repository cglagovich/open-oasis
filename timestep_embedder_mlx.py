import math
import mlx.core as mx
import mlx.nn as nn

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),  # hidden_size is diffusion model hidden size
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        # Create frequencies
        freqs = mx.exp(
            -math.log(max_period) * 
            mx.arange(start=0, stop=half, dtype=mx.float32) / half
        )
        
        # Expand dimensions for broadcasting
        t_expanded = mx.expand_dims(t.astype(mx.float32), axis=1)  # equivalent to t[:, None]
        freqs_expanded = mx.expand_dims(freqs, axis=0)  # equivalent to freqs[None]
        
        # Compute arguments for trig functions
        args = t_expanded * freqs_expanded
        
        # Compute embeddings
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=1)
        
        # Handle odd dimensions
        if dim % 2:
            zeros = mx.zeros_like(embedding[:, :1])
            embedding = mx.concatenate([embedding, zeros], axis=1)
            
        return embedding

    def __call__(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb