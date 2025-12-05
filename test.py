import torch, time

def linearInt(steps:torch.Tensor, corners:torch.Tensor):
    return corners[0]*(1.-steps)+corners[1]*steps

def bilinearInt(steps:torch.Tensor, corners:torch.Tensor):
    c12 = corners[0:2]
    c34 = corners[2:4]
    G12 = linearInt(steps[:,0],c12)
    G34 = linearInt(steps[:,0],c34)

    return G12*(1.-steps[:,1])+G34*steps[:,1]

def trilinearInt(steps:torch.Tensor, corners:torch.Tensor):
    c1234 = corners[0:4]
    c5678 = corners[4:8]
    G1234 = bilinearInt(steps[:,0:2],c1234)
    G5678 = bilinearInt(steps[:,0:2],c5678)

    return G1234*(1.-steps[:,2])+G5678*steps[:,2]


def trilinearInt_lowmem(steps: torch.Tensor, corners: torch.Tensor):
    """
    steps:   [N, 3]
    corners: [8, N, C]   (same as your code)

    Returns (N, C)
    """

    wx = steps[:, 0]
    wy = steps[:, 1]
    wz = steps[:, 2]

    wx1 = 1 - wx
    wy1 = 1 - wy
    wz1 = 1 - wz

    # Shorthands for readability
    c0 = corners[0]  # (N, C)
    c1 = corners[1]
    c2 = corners[2]
    c3 = corners[3]
    c4 = corners[4]
    c5 = corners[5]
    c6 = corners[6]
    c7 = corners[7]

    # Compute weighted sum directly (NO weight tensor)
    return (
            c0 * (wx1 * wy1 * wz1).unsqueeze(-1)
            + c1 * (wx * wy1 * wz1).unsqueeze(-1)
            + c2 * (wx1 * wy * wz1).unsqueeze(-1)
            + c3 * (wx * wy * wz1).unsqueeze(-1)
            + c4 * (wx1 * wy1 * wz).unsqueeze(-1)
            + c5 * (wx * wy1 * wz).unsqueeze(-1)
            + c6 * (wx1 * wy * wz).unsqueeze(-1)
            + c7 * (wx * wy * wz).unsqueeze(-1)
    )


def trilinearInt_fast(steps: torch.Tensor, corners: torch.Tensor):
    # steps: [N, 3]
    # corners: [8, N, C]

    wx = steps[:, 0]
    wy = steps[:, 1]
    wz = steps[:, 2]

    wx1 = 1 - wx
    wy1 = 1 - wy
    wz1 = 1 - wz

    # Compute weights (N,)
    w000 = wx1 * wy1 * wz1
    w100 = wx  * wy1 * wz1
    w010 = wx1 * wy  * wz1
    w110 = wx  * wy  * wz1
    w001 = wx1 * wy1 * wz
    w101 = wx  * wy1 * wz
    w011 = wx1 * wy  * wz
    w111 = wx  * wy  * wz

    weights = torch.stack([w000,w100,w010,w110,w001,w101,w011,w111], dim=0)  # [8, N]

    # Expand for broadcast: [8, N, 1] if corners has channels
    weights = weights.unsqueeze(-1)

    # Weighted sum
    result = (weights * corners).sum(dim=0)

    return result


def benchmark(fn, steps, corners):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    t0 = time.time()
    out = fn(steps, corners)
    torch.cuda.synchronize()
    t1 = time.time()

    mem = torch.cuda.max_memory_allocated() / (1024**2)
    return (t1 - t0) * 1000, mem, out


# Dummy data
N = 2_000_000
C = 4
steps = torch.rand((N, 3), device="cuda")
corners = torch.rand((8, N, C), device="cuda")


# New
print("Fast...")
t1, m1, o1 = benchmark(trilinearInt_lowmem, steps, corners)
print(f"Fast: {t1:.2f} ms, {m1:.1f} MB")

# # Original
# print("Original...")
# t0, m0, o0 = benchmark(trilinearInt, steps.unsqueeze(-1), corners)
# print(f"Original: {t0:.2f} ms, {m0:.1f} MB")

# # Check difference
# print("Max diff:", (o0 - o1).abs().max().item())
