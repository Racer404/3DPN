import os
import random
from typing import List, Any

import cv2
import numpy
import torch
from PIL import Image
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
        dSteps: int = 9,
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

    mse_loss = torch.nn.MSELoss()
    mae_loss = torch.nn.L1Loss()
    ssim_loss = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3)

    frames = []
    totalLoss = []
    dAlpha = utils.smoothStepsFunc(dSteps).to(device=device)

    for iter in range(iterations):
        random.shuffle(cameras)
        for cam in cameras:
            p_close, p_far = cam.getDepthRange(perlins[0].center, perlins[0].scale)
            d_start = p_close if p_close > 0. else 0.00001
            d_end = d_start + perlins[0].scale * 1.73205  # 1.73205 ~ sqrt(3)

            requestPoints_Volume = cam.sampleVolumeBySteps(d_start, d_end, dSteps)[0]
            mask_Volume = utils.maskValidPoints(requestPoints_Volume, perlins[0].center, perlins[0].scale)

            rendered_perPerlin = torch.stack([p.getValue(requestPoints_Volume[mask_Volume]) for p in perlins]) / 2. + 0.5 #Direct Scale
            rendered_perPerlin_color = rendered_perPerlin[:,:,0:perlins[0].channelNum - 1]
            rendered_perPerlin_alpha = rendered_perPerlin[:,:,perlins[0].channelNum - 1:perlins[0].channelNum]
            renderedPoints_Valid = (rendered_perPerlin_color * rendered_perPerlin_alpha).mean(dim=0)

            renderedPoints_Volume = torch.zeros([requestPoints_Volume.shape[0], perlins[0].channelNum - 1], dtype=torch.float64, device=device)
            renderedPoints_Volume[mask_Volume] = renderedPoints_Valid
            renderedPoints_Volume[~mask_Volume] = 0.

            renderedPoints_Flat = renderedPoints_Volume.reshape(cam.width * cam.height, dSteps, perlins[0].channelNum-1)
            # renderedPoints = renderedPoints_Flat.mean(dim=1)
            renderedPoints = torch.matmul(renderedPoints_Flat.transpose(1, 2), dAlpha)
            pred_img = renderedPoints.reshape(cam.width, cam.height, perlins[0].channelNum-1)

            mask_Flat = torch.any(mask_Volume.reshape(cam.width * cam.height, dSteps), dim=1)
            mask_img = mask_Flat.reshape(cam.width, cam.height)

            gtImage = (torch.tensor(cv2.imread(cam.image,cv2.IMREAD_COLOR_RGB), dtype=torch.float64, device=device)/255.).transpose(0,1)

            pred_img[~mask_img] = gtImage[~mask_img]
            loss = 0.9 * mse_loss(pred_img, gtImage) + 0.1 * (1-ssim_loss(pred_img.unsqueeze(0).permute(0,3,2,1), gtImage.unsqueeze(0).permute(0,3,2,1)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")
            totalLoss.append(loss.item())

            if ifVisualize:
                # renderedPoints[~output_mask] = ? #Background
                torch.cuda.synchronize()
                showImg = pred_img.transpose(0, 1).contiguous().cpu().detach().numpy()
                showGt = gtImage.transpose(0,1).cpu().detach().numpy()
                cv2.imshow("Training", showImg)
                cv2.imshow("GT", showGt)
                cv2.waitKey(1)

            if ifSaveGif:
                if cam is ref_cams[10]:
                    saveImg = (pred_img.transpose(0, 1).contiguous().cpu().detach().numpy() * 255).astype(numpy.uint8)
                    frames.append(saveImg)

            if iter is (iterations-1):
                if cam is ref_cams[10]:
                    saveImg = (pred_img.transpose(0, 1).contiguous().cpu().detach().numpy() * 255).astype(numpy.uint8)
                    saveGt = (gtImage.transpose(0,1).cpu().detach().numpy() * 255).astype(numpy.uint8)
                    cv2.imwrite(f"{resultFolder}/pred.png", cv2.cvtColor(saveImg, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(f"{resultFolder}/gt.png", cv2.cvtColor(saveGt, cv2.COLOR_RGB2BGR))

    if ifSaveGif:
        gif = [Image.fromarray(frame, mode="RGB") for frame in frames]
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

    for idx,p in enumerate(perlins):
        p.cornerVecs.requires_grad_(False)
        p.writeTensor(f"{resultFolder}/{str(idx)}.pth")

    return totalLoss

if __name__ == "__main__":
    dataset = "kitchen"
    cams = utils.readColmapSceneInfo(dataset)
    optimalZ = utils.getDOIfromCams(cams)
    sceneCenter, centerVar = utils.getPOIfromCamsZ(cams, optimalZ)

    trainingSetup = "test"
    outputFolder = f"{dataset}/trained/{trainingSetup}"

    p3 = PerlinNoise3D(scale=2, res=3, center=sceneCenter, channelNum=4, device="cuda")
    # p10 = PerlinNoise3D(scale=2, res=10, center=sceneCenter, channelNum=4, device="cuda")
    # p30 = PerlinNoise3D(scale=2, res=30, center=sceneCenter, channelNum=4, device="cuda")

    loss = train([p3], cams, 10, 0.01, 9, False, True, outputFolder)
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