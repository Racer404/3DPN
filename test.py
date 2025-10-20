import random

import torch
import time

# setup
N = 10_000_000
M_min = 8_000_000
M_max = 9_000_000

tensor_size_gb = 8  # Target 8GB, adjust based on your GPU's VRAM
num_elements = int(tensor_size_gb * (1024**3) / 4) # For float32


for i in range(1000):
    i=i+1
    A = torch.zeros(N, dtype=torch.float64)
    M = random.randint(M_min, M_max)
    B = torch.rand(M, dtype=torch.float64)
    indices = torch.randperm(N)[:M]  # random unique indices for B placement
    mask = torch.zeros(N, dtype=torch.bool)
    mask[indices] = True

    start = time.time()
    A[mask] = B
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    print(f"Boolean mask indexing: {time.time() - start:.4f} s")