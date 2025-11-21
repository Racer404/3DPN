import os
import random
from typing import List, Any

import cv2
import numpy
import torch
from matplotlib import pyplot as plt
from pytorch_msssim import SSIM
from torch import optim

import utils
from learnablePerlin3D import PerlinNoise3D


def train(
        perlins: List[PerlinNoise3D] = None,
        cameras: List[utils.Camera] = None,
        iterations: int = None,
        lr: float = None,
        ifVisualize: bool = False,
        ifSaveGif: bool = False,
        resultFolder: str = "results",
        device: str = "cuda") -> List[Any]:

    os.makedirs(resultFolder, exist_ok=True)

    for p in perlins:
        p.cornerVecs.requires_grad_(True)
    optimizer = optim.Adam(
        [p.cornerVecs for p in perlins], lr
    )

    mse_loss = torch.nn.MSELoss()
    mae_loss = torch.nn.L1Loss()
    bceLogit_loss = torch.nn.BCEWithLogitsLoss()
    ssim_loss = SSIM(win_size=11, win_sigma=1.5, data_range=1., size_average=True, channel=1)

    frames = []
    totalLoss = []
    dSteps = 100
    dAlpha = utils.smoothStepsFunc(dSteps).to(device=device)

    for iter in range(iterations):
        random.shuffle(cams)
        for cam in cameras:
            p_close, p_far = cam.getDepthRange(perlins[0].center, perlins[0].scale)
            d_start = p_close if p_close > 0. else 0.00001
            d_end = d_start + perlins[0].scale * 1.73205  # 1.73205 ~ sqrt(3)

            requestPoints_Volume = cam.sampleVolumeBySteps(d_start, d_end, dSteps)[0]
            mask_Volume = utils.maskValidPoints(requestPoints_Volume, perlins[0].center, perlins[0].scale)

            rendered_perPerlin = torch.stack([p.getValue(requestPoints_Volume[mask_Volume]) for p in perlins])
            renderedPoints_Valid = rendered_perPerlin.mean(dim=0)

            renderedPoints_Volume = torch.zeros(requestPoints_Volume.shape[0], dtype=torch.float64, device=device)
            renderedPoints_Volume[mask_Volume] = renderedPoints_Valid / 2. + 0.5
            renderedPoints_Volume[~mask_Volume] = 0.5

            renderedPoints_Flat = renderedPoints_Volume.reshape(cam.width * cam.height, dSteps)
            renderedPoints = renderedPoints_Flat @ dAlpha

            pred_img = renderedPoints.reshape(cam.width, cam.height)
            mask_Flat = torch.any(mask_Volume.reshape(cam.width * cam.height, dSteps), dim=1)
            mask_img = mask_Flat.reshape(cam.width, cam.height)

            gtImage = (torch.tensor(cv2.imread(cam.image,cv2.IMREAD_GRAYSCALE), dtype=torch.float64, device=device)/255.).T

            pred_img[~mask_img] = gtImage[~mask_img]
            loss_ssim = 1 - ssim_loss(pred_img.unsqueeze(0).unsqueeze(0).permute(0,1,3,2), gtImage.unsqueeze(0).unsqueeze(0).permute(0,1,3,2))
            loss_mse = mse_loss(pred_img, gtImage)
            loss = 0.1 * loss_ssim + 0.9 * loss_mse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")
            totalLoss.append(loss.item())

            if ifVisualize:
                # renderedPoints[~output_mask] = ? #Background
                torch.cuda.synchronize()
                showImg = pred_img.T.cpu().detach().numpy()
                showGt = gtImage.T.cpu().detach().numpy()
                cv2.imshow("Training", showImg)
                cv2.imshow("GT", showGt)
                cv2.waitKey(1)

        if ifSaveGif:
            print("UNDER CONSTRUCTION")
            # pred_img, mask = renderfromCamRay(resultRefCam, perlins, dSteps, dAlpha)
            # torch.cuda.synchronize()
            # frames.append((pred_img.T.cpu().detach().numpy() * 255).astype(numpy.uint8))

    if ifSaveGif:
        print("UNDER CONSTRUCTION")
        # gif = [Image.fromarray(frame) for frame in frames]
        # gif[0].save(
        #     f"{resultFolder}/training.gif",
        #     save_all=True,
        #     append_images=gif[1:],
        #     optimize=False,
        #     duration=1,
        #     loop=0,
        # )

    for idx,p in enumerate(perlins):
        p.cornerVecs.requires_grad_(False)
        p.writeTensor(f"{resultFolder}/{str(idx)}.pth")

    return totalLoss

if __name__ == "__main__":
    dataset = "kitchen"
    trainingSetup = "test"
    outputFolder = f"{dataset}/trained/{trainingSetup}"
    sceneCenter = torch.tensor([-0.461083, 1.5, 1.5], dtype=torch.float64, device="cuda")

    cams = utils.readColmapSceneInfo(dataset)
    perlin = PerlinNoise3D(scale=2, res=30, center=sceneCenter, device="cuda")

    loss = train([perlin], cams, 100, 0.01, True, False, outputFolder)

    loss_arr = numpy.array(loss)
    loss_arr = loss_arr.reshape([-1,len(cams)])
    loss_per_batch = loss_arr.mean(axis=1)
    torch.cuda.synchronize()
    ## END OF TRAINING



    # === Save histograms and plots ===
    def save_plot(fig_name):
        """Helper function to save current matplotlib figure."""
        plt.savefig(os.path.join(outputFolder, fig_name))
        plt.close()

    plt.figure()
    plt.plot(loss_per_batch)
    plt.annotate(str(loss_per_batch[-1]), xy=(len(loss_per_batch) - 1, loss_per_batch[-1]))
    save_plot("loss_batch.png")