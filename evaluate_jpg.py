import os
import torch

import lpips
import numpy as np
from PIL import Image
from tqdm import tqdm

# =========================
# Metrics
# =========================

def psnr_torch(pred, gt, eps=1e-8):
    """
    pred, gt: torch.Tensor, shape (H, W, 3), range [0,1]
    """
    mse = torch.mean((pred - gt) ** 2)
    return 10.0 * torch.log10(1.0 / (mse + eps))


lpips_fn = lpips.LPIPS(net='alex')
lpips_fn.eval()

def lpips_torch_alex(pred, gt):
    """
    pred, gt: torch.Tensor, shape (H, W, 3), range [0,1]
    """
    # HWC -> NCHW
    pred = pred.permute(2, 0, 1).unsqueeze(0)
    gt   = gt.permute(2, 0, 1).unsqueeze(0)

    # [0,1] -> [-1,1]
    pred = pred * 2.0 - 1.0
    gt   = gt * 2.0 - 1.0

    with torch.no_grad():
        val = lpips_fn(pred, gt)

    return val.item()


def ssim_torch(pred, gt, window_size=11, sigma=1.5, eps=1e-8):
    """
    pred, gt: torch.Tensor
        Shape (H, W, 3)
        Range [0,1]
    Returns:
        SSIM scalar
    """
    # HWC -> NCHW
    pred = pred.permute(2, 0, 1).unsqueeze(0)
    gt   = gt.permute(2, 0, 1).unsqueeze(0)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Gaussian window
    coords = torch.arange(window_size, device=pred.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).view(1, 1, -1)

    window = g.transpose(2, 1) @ g
    window = window.expand(3, 1, window_size, window_size)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=3)
    mu2 = F.conv2d(gt,   window, padding=window_size // 2, groups=3)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(gt * gt,     window, padding=window_size // 2, groups=3) - mu2_sq
    sigma12   = F.conv2d(pred * gt,   window, padding=window_size // 2, groups=3) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + eps
    )

    return ssim_map.mean()

# =========================
# Image loading
# =========================

def load_image(path):
    """
    Returns torch.Tensor of shape (H, W, 3), range [0,1]
    """
    img = Image.open(path).convert("RGB")
    img = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(img)


# =========================
# Main evaluation
# =========================

gt_dir   = "/path/to/groundtruth"
pred_dir = "scale=2_res=64_dSteps=128_decay_bg=0.5_mae.8+ssim.2"

psnr_list  = []
lpips_list = []

# Only evaluate files that exist in BOTH folders
filenames = sorted([
    f for f in os.listdir(gt_dir)
    if f.endswith(".JPG") and os.path.exists(os.path.join(pred_dir, f))
])

for fname in tqdm(filenames):
    gt_path   = os.path.join(gt_dir, fname)
    pred_path = os.path.join(pred_dir, fname)

    gt   = load_image(gt_path)
    pred = load_image(pred_path)

    assert gt.shape == pred.shape, f"Shape mismatch: {fname}"

    psnr_val  = psnr_torch(pred, gt).item()
    lpips_val = lpips_torch_alex(pred, gt)

    psnr_list.append(psnr_val)
    lpips_list.append(lpips_val)


# =========================
# Results
# =========================

print(f"Images evaluated: {len(psnr_list)}")
print(f"Mean PSNR : {np.mean(psnr_list):.4f}")
print(f"Mean LPIPS: {np.mean(lpips_list):.4f}")