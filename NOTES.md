Reference impl on M1:
Takes between 129s and 1114s per iteration. Later iterations are slower because DiT is recomputing for previous frames.

 
19%|██████████████████████████▎                                                                                                             | 6/31 [10:50<53:46, 129.05s/it]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31/31 [9:35:47<00:00, 1114.45s/it]


## Benchmarking