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

    ref_cams = cameras[:]
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

    if ifSaveGif:
        for i in range(iterations):
            os.makedirs(f"{resultFolder}/iters/{i}", exist_ok=True)

    for iter in range(iterations):
        random.shuffle(cameras)
        for cam in cameras:
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
            mask_img = mask_Flat.reshape(cam.width, cam.height)

            gtImage = (torch.tensor(cv2.imread(cam.image,cv2.IMREAD_COLOR_RGB), dtype=torch.float32, device=device)/255.).transpose(0,1)
            pred_img[~mask_img] = gtImage[~mask_img]
            loss = 0.8 * mae_loss(pred_img, gtImage) + 0.2 * (1-ssim_loss(pred_img.unsqueeze(0).permute(0,3,2,1), gtImage.unsqueeze(0).permute(0,3,2,1)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")
            totalLoss.append(loss.item())

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
                cv2.imwrite(f"{resultFolder}/iters/{iter}/{os.path.basename(cam.image)}", cv2.cvtColor(pred_numpy, cv2.COLOR_RGB2BGR))

    for idx,p in enumerate(perlins):
        p.cornerVecs.requires_grad_(False)
        os.makedirs(f"{resultFolder}/weights", exist_ok=True)
        p.writeTensor(f"{resultFolder}/weights/{str(idx)}.pth")

    return totalLoss

if __name__ == "__main__":
    dataset = "bonsai"
    cams = utils.readColmapSceneInfo(dataset)
    optimalZ = utils.getDOIfromCams(cams)
    sceneCenter, centerVar = utils.getPOIfromCamsZ(cams, optimalZ)

    targetRes = {59}

    for res in targetRes:
        trainingSetup = f"scale=2_res={res}_dSteps={2 * res}_decay_bg=0.5_mae.8+ssim.2"
        outputFolder = f"{dataset}/trained/{trainingSetup}"
        perlin = PerlinNoise3D(scale=2, res=res, center=sceneCenter, channelNum=4, device="cuda")
        loss = train([perlin], cams, 100, 0.01, 2 * res, True, False, outputFolder)

        loss_arr = numpy.array(loss)
        loss_arr = loss_arr.reshape([-1,len(cams)])
        loss_per_batch = loss_arr.mean(axis=1)
        torch.cuda.synchronize()
        ## END OF TRAINING

        plt.figure()
        plt.plot(loss_per_batch)
        plt.annotate(str(loss_per_batch[-1]), xy=(len(loss_per_batch) - 1, loss_per_batch[-1]))
        plt.savefig(f"{outputFolder}/loss_batch.png")
        plt.close()
        numpy.savetxt(f"{outputFolder}/loss.txt", loss_arr)
        numpy.savetxt(f"{outputFolder}/loss_batch.txt", loss_per_batch)