# Fork info
This is my fork of open-oasis to add MLX support! Follow all normal instructions and run on mlx with the following:
```
python generate_mlx.py
```
So far this is a pretty basic MLX port - I'm sure there are some interesting perf optimizations I'm leaving on the table. Beyond those unknown MLX optimizations and quantization, I'm very interested in KV caching. You can see generation slow down dramatically with frame number which should be avoidable with caching. 

### Timing results:
The MLX version can be almost 5x faster than torch on CPU. I'm measuring on my 2020 M1 Macbook Pro.

| backend | end-to-end generate 5 frames |
|------------|-------|
| torch cpu | 149.56 s |
| mlx | 32.08 s |

## Future work
- [ ] Get time on M4 macbook
- [ ] Add KV caching

# Oasis 500M

Oasis is an interactive world model developed by [Decart](https://www.decart.ai/) and [Etched](https://www.etched.com/). Based on diffusion transformers, Oasis takes in user keyboard input and generates gameplay in an autoregressive manner. We release the weights for Oasis 500M, a downscaled version of the model, along with inference code for action-conditional frame generation. 

For more details, see our [joint blog post](https://oasis-model.github.io/) to learn more.

And to use the most powerful version of the model, be sure to check out the [live demo](https://oasis.us.decart.ai/) as well!

## Setup
```
git clone https://github.com/etched-ai/open-oasis.git
cd open-oasis
# Install pytorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Install other dependencies
pip install einops diffusers timm av
```

## Download the model weights
Inside the `open-oasis/` directory, run:
```
huggingface-cli login
huggingface-cli download Etched/oasis-500m oasis500m.safetensors # DiT checkpoint
huggingface-cli download Etched/oasis-500m vit-l-20.safetensors  # ViT VAE checkpoint
```

## Basic Usage
We include a basic inference script that loads a prompt frame from a video and generates additional frames conditioned on actions.
```
python generate.py
# Or specify path to checkpoints:
python generate.py --oasis-ckpt <path to oasis500m.safetensors> --vae-ckpt <path to vit-l-20.safetensors>
```
Use a custom image prompt:
```
python generate.py --prompt-path <path to .png, .jpg, or .jpeg>
```
