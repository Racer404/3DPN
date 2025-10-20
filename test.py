import torch
import time

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Large example: [8, N, 3]
N = 10_000_000
M = 1_000_000
corner_Flat = torch.rand(8, N, 3, device=device)
idx = torch.randint(0, N, (M,), device=device)

# -------------------------
# Method 1: index_select
# -------------------------
torch.cuda.synchronize() if device=="cuda" else None
start = time.time()
corner_chunk1 = torch.index_select(corner_Flat, 1, idx)

torch.cuda.synchronize() if device=="cuda" else None
t1 = time.time() - start
print(f"index_select time: {t1:.4f}s, output shape: {corner_chunk1.shape}")

# -------------------------
# Method 2: take_along_dim
# -------------------------
idx_expand = idx.view(1, -1, 1).expand(8, -1, 3)

torch.cuda.synchronize() if device=="cuda" else None
start = time.time()

corner_chunk2 = torch.take_along_dim(corner_Flat, idx_expand, dim=1)

torch.cuda.synchronize() if device=="cuda" else None
t2 = time.time() - start
print(f"take_along_dim time: {t2:.4f}s, output shape: {corner_chunk2.shape}")

# Verify they are equal
print("Equal:", torch.allclose(corner_chunk1, corner_chunk2))
