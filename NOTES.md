Reference impl on M1:
Takes between 129s and 1114s per iteration. Later iterations are slower because DiT is recomputing for previous frames.

 
19%|██████████████████████████▎                                                                                                             | 6/31 [10:50<53:46, 129.05s/it]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31/31 [9:35:47<00:00, 1114.45s/it]


## Benchmarking
Device mps
DiT: 340.14 ms

Device cpu
DiT: 544.79 ms

MPS is faster by a bit. 

First optimization will be to rewrite in MLX. 
- [ ] Convert pt state_dict to mlx
- [ ] Test modules
    - [x] ROPE
    - [x] MLP
    - [x] ATTN
    - [x] DIT BLOCK
    - [x] DIT
    - [ ] VAE

MLX DiT is 4.29s per iteration on 4th iteration
Torch DiT is 9.02s per iteration on 4th iteration

Add KV caching -- past frames only needed for temporal attention

# Commands
```
python generate.py --oasis-ckpt oasis500m.pt --vae-ckpt vit-l-20.pt 


```