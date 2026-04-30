import math
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
        perlin: PerlinNoise3D = None,
        cameras: List[utils.Camera] = None,
        iterations: int = None,
        lr: float = None,
        optimalZ: float = None,
        dSteps: int = None,
        ifVisualize: bool = False,
        ifSaveGif: bool = False,
        resultFolder: str = "results",
        device: str = "cuda") -> List[Any]:

    os.makedirs(resultFolder, exist_ok=True)

    perlin.cornerVecs.requires_grad_(True)
    optimizer = optim.Adam(
        [perlin.cornerVecs], lr
    )

    # mse_loss = torch.nn.MSELoss()
    mae_loss = torch.nn.L1Loss()
    ssim_loss = SSIM(win_size=11, win_sigma=1.5, data_range=1., size_average=True, channel=3)

    totalLoss = []

    epochs = iterations/len(cams)
    epochs_full = math.floor(epochs)

    if ifSaveGif:
        for i in range(epochs_full+1):
            os.makedirs(f"{resultFolder}/epochs/{i}", exist_ok=True)

    reminderCamNum = math.floor(len(cameras)*(epochs-epochs_full))
    for epoch in range(epochs_full+1):
        random.shuffle(cameras)

        if epoch == epochs_full:
            cameras = cameras[:reminderCamNum]

        loss_epoch = []
        for cam in cameras:
            p_close = 0.
            p_far = optimalZ * 2
            requestPoints_Volume = cam.sampleVolumeBySteps(p_close, p_far, dSteps)[0]
            renderedPoints_Volume = perlin.getValue(requestPoints_Volume, optimizer) / 2. + 0.5

            rendered_color = renderedPoints_Volume[:, :-1]
            rendered_alpha = renderedPoints_Volume[:, -1]

            renderedPoints_Flat = utils.renderVolume_stepsRaypass(rendered_color,rendered_alpha,dSteps).squeeze()

            pred_img = renderedPoints_Flat.reshape(cam.width, cam.height, 3)
            gtImage = (torch.tensor(cv2.imread(cam.image,cv2.IMREAD_COLOR_RGB), dtype=torch.float32, device=device)/255.).transpose(0,1)

            loss_ssim = 1 - ssim_loss(pred_img.unsqueeze(0).permute(0,3,2,1), gtImage.unsqueeze(0).permute(0,3,2,1))
            loss_mae = mae_loss(pred_img, gtImage)
            loss = 0.2 * loss_ssim + 0.8 * loss_mae

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epochs {epoch + 1}/{epochs_full+1}, Loss: {loss.item()}")
            loss_epoch.append(loss.item())

            if ifVisualize:
                torch.cuda.synchronize()
                pred_numpy = pred_img.transpose(0, 1).cpu().detach().numpy()
                pred_numpy = numpy.clip(pred_numpy, 0., 1.)
                showGt = gtImage.transpose(0, 1).cpu().detach().numpy()
                cv2.imshow("Training", pred_numpy)
                cv2.imshow("GT", showGt)
                cv2.waitKey(1)

            if ifSaveGif:
                torch.cuda.synchronize()
                pred_numpy = pred_img.transpose(0, 1).cpu().detach().numpy()
                pred_numpy = numpy.clip(pred_numpy, 0., 1.)
                pred_numpy = (pred_numpy * 255).astype(numpy.uint8)
                cv2.imwrite(f"{resultFolder}/epochs/{epoch}/{os.path.basename(cam.image)}", cv2.cvtColor(pred_numpy, cv2.COLOR_RGB2BGR))
        totalLoss.append(loss_epoch)

    perlin.cornerVecs.requires_grad_(False)
    os.makedirs(f"{resultFolder}/weights", exist_ok=True)
    perlin.writeTensor(f"{resultFolder}/weights/0.pth")

    return totalLoss

if __name__ == "__main__":
    datasets = ["room"]
    targetRes = [7]

    for res in targetRes:
        for dataset in datasets:
            cams = utils.readColmapSceneInfo(f"data/{dataset}")
            optimalZ = utils.getDOIfromCams(cams)
            sceneCenter, centerVar = utils.getPOIfromCamsZ(cams, optimalZ)
            nyquist_freq = math.ceil(2 * (res * optimalZ * 2))

            trainingSetup = f"INF_res={res}_dStep={nyquist_freq}_raypass_mae.8+ssim.2"
            outputFolder = f"data/{dataset}/trained/{trainingSetup}"

            perlin = PerlinNoise3D(res=res, center=sceneCenter, channelNum=3+1, device="cuda")
            loss = train(perlin, cams, 20_000, 0.01, optimalZ, nyquist_freq, True, True, outputFolder)
            torch.cuda.synchronize()
            ## END OF TRAINING

            loss_per_epoch = numpy.array([numpy.mean(i) for i in loss])

            plt.figure()
            plt.plot(loss_per_epoch)
            plt.annotate(str(loss_per_epoch[-1]), xy=(len(loss_per_epoch) - 1, loss_per_epoch[-1]))
            plt.savefig(f"{outputFolder}/loss_epoch.png")
            plt.close()