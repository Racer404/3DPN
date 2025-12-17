import os
import time
from typing import List, Any, Tuple

import cv2
import lpips
import numpy
import torch
import torch.nn.functional as F
from tqdm import tqdm

import learnablePerlin3D
import utils
from learnablePerlin3D import PerlinNoise3D

# Initialize once
lpips_fn = lpips.LPIPS(net='alex').to("cuda")
lpips_fn.eval()

def lpips_torch(pred, gt):
    assert pred.shape == gt.shape
    assert pred.ndim == 3 and pred.shape[-1] == 3

    # HWC -> NCHW
    pred = pred.permute(2, 0, 1).unsqueeze(0)
    gt   = gt.permute(2, 0, 1).unsqueeze(0)

    # Normalize to [-1, 1]
    pred = pred * 2.0 - 1.0
    gt   = gt * 2.0 - 1.0

    with torch.no_grad():
        lpips_val = lpips_fn(pred, gt)

    return lpips_val.item()

def psnr_torch(pred, gt, eps=1e-8):
    mse = torch.mean((pred - gt) ** 2)
    psnr = 10.0 * torch.log10(1.0 / (mse + eps))
    return psnr

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

def eval(
        perlins: List[PerlinNoise3D] = None,
        cameras: List[utils.Camera] = None,
        dSteps: int = None,
        ifVisualize: bool = False,
        extremeSpeed: bool = False,
        device: str = "cuda") -> Tuple[List[Any], List[Any], List[Any]]:

    totalLoss_psnr = []
    totalLoss_lpips = []
    totalLoss_ssim = []

    center_ = perlins[0].center
    scale_ = perlins[0].scale
    colorChannels_ = perlins[0].channelNum-1

    if extremeSpeed:
        for cam in tqdm(cameras):
            p_close, p_far = cam.getDepthRange(center_, scale_)
            d_start = p_close if p_close > 0. else 0.00001
            d_end = d_start + scale_ * 1.73205  # 1.73205 ~ sqrt(3)

            requestPoints_Volume = cam.sampleVolumeBySteps(d_start, d_end, dSteps)[0]
            mask_Volume = utils.maskValidPoints(requestPoints_Volume, center_, scale_)

            rendered_perPerlin = torch.stack([p.getValue(requestPoints_Volume[mask_Volume]) for p in perlins]) / 2. + 0.5 #Direct Scale
            rendered_perPerlin_color = rendered_perPerlin[:,:,:-1]
            rendered_perPerlin_alpha = rendered_perPerlin[:,:,-1:]
            renderedPoints_Flat, mask_Flat = utils.renderVolume_stepsDecay(rendered_perPerlin_color, rendered_perPerlin_alpha, mask_Volume, dSteps)

            pred_img = renderedPoints_Flat.reshape(cam.width, cam.height, colorChannels_)
            pred_img = torch.clip(pred_img, 0, 1)

            if ifVisualize:
                torch.cuda.synchronize()
                pred_numpy = pred_img.transpose(0, 1).contiguous().cpu().detach().numpy()
                cv2.imshow("Training", pred_numpy)
                cv2.waitKey(1)
        pass
    else:
        for cam in tqdm(cameras):
            p_close, p_far = cam.getDepthRange(center_, scale_)
            d_start = p_close if p_close > 0. else 0.00001
            d_end = d_start + scale_ * 1.73205  # 1.73205 ~ sqrt(3)

            requestPoints_Volume = cam.sampleVolumeBySteps(d_start, d_end, dSteps)[0]
            mask_Volume = utils.maskValidPoints(requestPoints_Volume, center_, scale_)

            rendered_perPerlin = torch.stack([p.getValue(requestPoints_Volume[mask_Volume]) for p in perlins]) / 2. + 0.5 #Direct Scale
            rendered_perPerlin_color = rendered_perPerlin[:,:,:-1]
            rendered_perPerlin_alpha = rendered_perPerlin[:,:,-1:]
            renderedPoints_Flat, mask_Flat = utils.renderVolume_stepsDecay(rendered_perPerlin_color, rendered_perPerlin_alpha, mask_Volume, dSteps)

            pred_img = renderedPoints_Flat.reshape(cam.width, cam.height, colorChannels_)
            pred_img = torch.clip(pred_img, 0, 1)
            mask_img = mask_Flat.reshape(cam.width, cam.height)

            gtImage = (torch.tensor(cv2.imread(cam.image,cv2.IMREAD_COLOR_RGB), dtype=torch.float32, device=device)/255.).transpose(0,1)
            gtImage[~mask_img] = 0.
            pred_img[~mask_img] = 0.
            # pred_img[~mask_img] = gtImage[~mask_img]

            totalLoss_psnr.append(psnr_torch(pred_img, gtImage).item())
            totalLoss_lpips.append(lpips_torch(pred_img, gtImage))
            totalLoss_ssim.append(ssim_torch(pred_img, gtImage).item())

            if ifVisualize:
                # pred_img[~mask_img] = ? #Background
                torch.cuda.synchronize()
                pred_numpy = pred_img.transpose(0, 1).contiguous().cpu().detach().numpy()
                showGt = gtImage.transpose(0,1).cpu().detach().numpy()
                cv2.imshow("Training", pred_numpy)
                cv2.imshow("GT", showGt)
                cv2.waitKey(1)

    return totalLoss_psnr, totalLoss_lpips, totalLoss_ssim

if __name__ == "__main__":
    datasets = ["plant","garden"]
    res = [64, 16, 4]

    for dataset in datasets:
        print(f"current scene:{dataset}")
        cams = utils.readColmapSceneInfo(dataset)
        optimalZ = utils.getDOIfromCams(cams)
        sceneCenter, centerStd = utils.getPOIfromCamsZ(cams, optimalZ)
        print(f"scene optimalZ :{optimalZ}")
        print(f"scene center :{sceneCenter}")
        print(f"scene scale std:{centerStd}")
        trainingSetup = f"scale=4.25_res={res[0]}+{res[1]}+{res[2]}_dSteps={2 * res[0]}_decay_bg=0.5_mae.8+ssim.2"

        perlinFolder = f"{dataset}/trained/{trainingSetup}/weights"
        outputFolder = f"{dataset}/trained/{trainingSetup}/eval"
        os.makedirs(outputFolder, exist_ok=True)

        perlin1 = learnablePerlin3D.readTensor(f"{perlinFolder}/0.pth")
        perlin2 = learnablePerlin3D.readTensor(f"{perlinFolder}/1.pth")
        perlin3 = learnablePerlin3D.readTensor(f"{perlinFolder}/2.pth")

        nyquistFreq = res[0] * 2

        start = time.time()
        loss_psnr, loss_lpips, loss_ssim = eval([perlin1,perlin2,perlin3], cams, nyquistFreq, False, False, "cuda")
        totalTime = time.time() - start

        print(f"fps:{len(cams)/totalTime}")

        torch.cuda.synchronize()
        ## END OF TRAINING
        loss_psnr_arr = numpy.array(loss_psnr)
        loss_lpips_arr = numpy.array(loss_lpips)
        loss_ssim_arr = numpy.array(loss_ssim)
        print(f"mean_psnr:{loss_psnr_arr.mean()}")
        print(f"mean_lpips:{loss_lpips_arr.mean()}")
        print(f"mean_ssim:{loss_ssim_arr.mean()}")
        numpy.savetxt(f"{outputFolder}/loss_psnr.txt", loss_psnr_arr)
        numpy.savetxt(f"{outputFolder}/loss_lpips.txt", loss_lpips_arr)
        numpy.savetxt(f"{outputFolder}/loss_ssim.txt", loss_ssim_arr)