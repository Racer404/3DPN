import os
import random
from typing import List, Any

import cv2
import numpy
import torch
from PIL import Image
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
        dSteps: int = None,
        ifVisualize: bool = False,
        ifSaveGif: bool = False,
        resultFolder: str = "results",
        device: str = "cuda") -> List[Any]:

    ref_cams = cameras[:]
    os.makedirs(resultFolder, exist_ok=True)

    perlin.cornerVecs.requires_grad_(True)
    optimizer = optim.Adam(
        [perlin.cornerVecs], lr
    )

    mse_loss = torch.nn.MSELoss()
    mae_loss = torch.nn.L1Loss()
    bceLogit_loss = torch.nn.BCEWithLogitsLoss()
    ssim_loss = SSIM(win_size=11, win_sigma=1.5, data_range=1., size_average=True, channel=1)

    frames = []
    totalLoss = []
    dAlpha = utils.smoothStepsFunc(dSteps).to(device=device)

    for iter in range(iterations):
        random.shuffle(cams)
        for cam in cameras:
            p_close = 2.
            p_far = 4.5
            samplePoints_Volume = cam.sampleVolumeBySteps(p_close, p_far, dSteps)[0]
            renderedPoints_Volume = perlin.getValue(samplePoints_Volume, optimizer) / 2. + 0.5

            renderedPoints_Flat = renderedPoints_Volume.reshape(cam.width * cam.height, dSteps)
            renderedPoints = renderedPoints_Flat @ dAlpha

            pred_img = renderedPoints.reshape(cam.width, cam.height)

            gtImage = (torch.tensor(cv2.imread(cam.image,cv2.IMREAD_GRAYSCALE), dtype=torch.float64, device=device)/255.).T

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
                if cam is ref_cams[10]:
                    saveImg = (pred_img.T.cpu().detach().numpy() * 255).astype(numpy.uint8)
                    frames.append(saveImg)

            if iter is (iterations-1):
                if cam is ref_cams[10]:
                    saveImg = (pred_img.T.cpu().detach().numpy() * 255).astype(numpy.uint8)
                    saveGt = (gtImage.T.cpu().detach().numpy() * 255).astype(numpy.uint8)
                    cv2.imwrite(f"{resultFolder}/pred.png", cv2.cvtColor(saveImg, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(f"{resultFolder}/gt.png", cv2.cvtColor(saveGt, cv2.COLOR_RGB2BGR))

    if ifSaveGif:
        gif = [Image.fromarray(frame) for frame in frames]
        out_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(out_dir, exist_ok=True)
        gif[0].save(
            f"{resultFolder}/training.gif",
            save_all=True,
            append_images=gif[1:],
            optimize=False,
            duration=1,
            loop=0,
        )

    perlin.cornerVecs.requires_grad_(False)
    perlin.writeTensor(f"{resultFolder}/0.pth")

    return totalLoss

if __name__ == "__main__":
    dataset = "kitchen"
    trainingSetup = "test_INF"
    outputFolder = f"{dataset}/trained/{trainingSetup}"

    cams = utils.readColmapSceneInfo(dataset)
    optimalZ = utils.getDOIfromCams(cams)
    sceneCenter, centerVar = utils.getPOIfromCamsZ(cams, optimalZ)
    perlin = PerlinNoise3D(res=2, center=sceneCenter, device="cuda")

    loss = train(perlin, cams, 5, 0.01, 10, True, True, outputFolder)

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