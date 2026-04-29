import math
import os
import random
from typing import List, Any

import cv2
import numpy
import torch
from pytorch_msssim import SSIM
from torch import optim

import utils
from learnablePerlin3D import PerlinNoise3D
from matplotlib import pyplot as plt

def train(
        perlins: List[PerlinNoise3D] = None,
        cameras: List[utils.Camera] = None,
        iterations: int = None,
        lr: float = None,
        dSteps: int = None,
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

    # mse_loss = torch.nn.MSELoss()
    mae_loss = torch.nn.L1Loss()
    ssim_loss = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3)

    totalLoss = []

    center_ = perlins[0].center
    scale_ = perlins[0].scale
    colorChannels_ = perlins[0].channelNum-1

    epochs = iterations/len(cameras)
    epochs_full = math.floor(epochs)

    if ifSaveGif:
        for i in range(epochs_full+1):
            os.makedirs(f"{resultFolder}/epochs/{i}", exist_ok=True)

    reminderNum = math.floor(len(cameras) * (epochs-epochs_full))
    for epoch in range(epochs_full+1):
        random.shuffle(cameras)

        if epoch == epochs_full:
            cameras = cameras[:reminderNum]

        loss_epoch = []
        for cam in cameras:
            p_close, p_far = cam.getDepthRange(center_, scale_)
            d_start = p_close if p_close > 0. else 0.00001
            d_end = d_start + scale_ * 1.73205  # 1.73205 ~ sqrt(3)

            requestPoints_Volume = cam.sampleVolumeBySteps(d_start, d_end, dSteps)[0]
            mask_Volume = utils.maskValidPoints(requestPoints_Volume, center_, scale_)

            rendered_perPerlin = torch.stack([p.getValue(requestPoints_Volume[mask_Volume]) for p in perlins]) / 2. + 0.5 #Direct Scale
            rendered_perPerlin_color = rendered_perPerlin[:,:,:-1]
            rendered_perPerlin_alpha = rendered_perPerlin[:,:,-1:]
            renderedPoints_Flat, mask_Flat = utils.renderVolume_stepsRaypass(rendered_perPerlin_color, rendered_perPerlin_alpha, mask_Volume, dSteps)

            pred_img = renderedPoints_Flat.reshape(cam.width, cam.height, colorChannels_)
            mask_img = mask_Flat.reshape(cam.width, cam.height)

            gtImage = (torch.tensor(cv2.imread(cam.image,cv2.IMREAD_COLOR_RGB), dtype=torch.float32, device=device)/255.).transpose(0,1)
            pred_img[~mask_img] = gtImage[~mask_img]
            loss = 0.8 * mae_loss(pred_img, gtImage) + 0.2 * (1-ssim_loss(pred_img.unsqueeze(0).permute(0,3,2,1), gtImage.unsqueeze(0).permute(0,3,2,1)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epochs {epoch + 1}/{epochs_full+1}, Loss: {loss.item()}")
            loss_epoch.append(loss.item())

            if ifVisualize:
                # pred_img[~mask_img] = ? #Background
                torch.cuda.synchronize()
                pred_numpy = pred_img.transpose(0, 1).contiguous().cpu().detach().numpy()
                pred_numpy = numpy.clip(pred_numpy, 0., 1.)
                showGt = gtImage.transpose(0,1).cpu().detach().numpy()
                cv2.imshow("Training", pred_numpy)
                cv2.imshow("GT", showGt)
                cv2.waitKey(1)

            if ifSaveGif:
                pred_img[~mask_img] = 0. #Background
                pred_numpy = pred_img.transpose(0, 1).contiguous().cpu().detach().numpy()
                pred_numpy = numpy.clip(pred_numpy, 0., 1.)
                pred_numpy = (pred_numpy * 255).astype(numpy.uint8)
                cv2.imwrite(f"{resultFolder}/epochs/{epoch}/{os.path.basename(cam.image)}", cv2.cvtColor(pred_numpy, cv2.COLOR_RGB2BGR))
        totalLoss.append(loss_epoch)

    for idx,p in enumerate(perlins):
        p.cornerVecs.requires_grad_(False)
        os.makedirs(f"{resultFolder}/weights", exist_ok=True)
        p.writeTensor(f"{resultFolder}/weights/{str(idx)}.pth")

    return totalLoss

if __name__ == "__main__":
    datasets = ["room"]
    targetRes = [[64,16,4]]

    for res in targetRes:
        for dataset in datasets:
            cams = utils.readColmapSceneInfo(dataset)
            optimalZ = utils.getDOIfromCams(cams)
            sceneCenter, centerStd = utils.getPOIfromCamsZ(cams, optimalZ)
            scale_multiplier = 4.25
            print(f"scene centerStd:{centerStd}")
            trainingSetup = f"scale={scale_multiplier}_res={res[0]}+{res[1]}+{res[2]}_dSteps={2 * res[0]}_rayPass_bg=0.5_mae.8+ssim.2"
            outputFolder = f"{dataset}/trained/{trainingSetup}"
            perlin1 = PerlinNoise3D(scale=centerStd * scale_multiplier, res=res[0], center=sceneCenter, channelNum=4, device="cuda")
            perlin2 = PerlinNoise3D(scale=centerStd * scale_multiplier, res=res[1], center=sceneCenter, channelNum=4, device="cuda")
            perlin3 = PerlinNoise3D(scale=centerStd * scale_multiplier, res=res[2], center=sceneCenter, channelNum=4, device="cuda")

            loss = train([perlin1, perlin2, perlin3], cams, 20_000, 0.01, 2 * res[0], False, True, outputFolder)

            torch.cuda.synchronize()
            ## END OF TRAINING
            loss_per_epoch = numpy.array([numpy.mean(i) for i in loss])
            plt.figure()
            plt.plot(loss_per_epoch)
            plt.annotate(str(loss_per_epoch[-1]), xy=(len(loss_per_epoch) - 3, loss_per_epoch[-1]))
            plt.savefig(f"{outputFolder}/loss_epoch.png")
            plt.close()