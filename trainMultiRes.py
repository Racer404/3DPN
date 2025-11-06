import os
from typing import List, Any

import cv2
import torch
from PIL import Image
from torch import optim

import utils
from learnablePerlin3D import PerlinNoise3D
from matplotlib import pyplot as plt

def train(
        perlins: [PerlinNoise3D] = None,
        cameras: [utils.Camera] = None,
        iterations: int = None,
        lr: float = None,
        ifVisualize: bool = False,
        ifSaveResult: bool = False,
        resultTensorPth: str = "kitchen/trained/") -> List[Any]:

    for p in perlins:
        p.cornerVecs.requires_grad_(True)
    optimizer = optim.Adam(
        [p.cornerVecs for p in perlins], lr
    )

    mse_loss = torch.nn.MSELoss()
    mae_loss = torch.nn.L1Loss()

    frames = []
    totalLoss = []
    dSteps = 100
    dAlpha = utils.smoothStepsFunc(dSteps).to(device=cams[0].device)

    for iter in range(iterations):
        for cam in cameras:
            dClose, dFar = cam.getDepthRange(perlins[0])
            samplePoints_Volume, validPoints = cam.sampleVolumeBySteps(dClose, dFar, dSteps)

            output_mask_Volume = None
            renderedPoints_Volume = 0
            for p in perlins:
                renderedPoints_vol, output_mask_Volume = p.getValue(samplePoints_Volume, validPoints)
                renderedPoints_Volume = renderedPoints_Volume + renderedPoints_vol
            renderedPoints_Volume = renderedPoints_Volume / len(perlins)

            renderedPoints_Flat = renderedPoints_Volume.reshape(cam.width * cam.height, dSteps, perlins[0].channelNum)
            renderedPoints = torch.matmul(renderedPoints_Flat.transpose(1, 2), dAlpha)

            gtImage = (torch.tensor(cv2.imread(cam.image,cv2.IMREAD_COLOR_RGB), dtype=torch.float64, device="cuda")/255.).transpose(0,1)
            gtImage_Flat = gtImage.reshape(-1, perlins[0].channelNum)

            output_mask_Flat = output_mask_Volume.reshape(cam.width * cam.height, dSteps)
            output_mask = torch.any(output_mask_Flat, dim=1)

            loss = mse_loss(renderedPoints[output_mask], gtImage_Flat[output_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")
            totalLoss.append(loss.item())

            if ifVisualize:
                # renderedPoints[~output_mask] = ? #Background
                output_img = renderedPoints.reshape(cam.width, cam.height, perlins[0].channelNum)
                torch.cuda.synchronize()
                showImg = output_img.transpose(0, 1).contiguous().cpu().detach().numpy()
                showGt = gtImage.transpose(0,1).cpu().detach().numpy()
                cv2.imshow("Training", showImg)
                cv2.imshow("GT", showGt)
                cv2.waitKey(1)

                if ifSaveResult:
                    frames.append(showImg)

    if ifSaveResult:
        frames = [Image.fromarray(frame) for frame in frames]
        out_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(out_dir, exist_ok=True)
        frames[0].save(
            f"{out_dir}/training.gif",
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=1,
            loop=0,
        )

    for idx,p in enumerate(perlins):
        p.cornerVecs.requires_grad_(False)
        p.writeTensor(resultTensorPth+str(idx)+".pth")

    return totalLoss

if __name__ == "__main__":
    dataset = "kitchen"
    cams = utils.readColmapSceneInfo(dataset)
    testCenter = torch.tensor([-0.461083, 1.5, 1.5], dtype=torch.float64, device="cuda")

    p3 = PerlinNoise3D(scale=2, res=3, center=testCenter, channelNum=3, device="cuda")
    p10 = PerlinNoise3D(scale=2, res=10, center=testCenter, channelNum=3, device="cuda")
    p30 = PerlinNoise3D(scale=2, res=30, center=testCenter, channelNum=3, device="cuda")

    loss = train([p3,p10,p30], cams, 100, 0.01, True, False)
    torch.cuda.synchronize()
    plt.plot(loss)
    plt.show()