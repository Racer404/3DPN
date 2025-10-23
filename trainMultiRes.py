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

            renderedPoints_Flat = renderedPoints_Volume.reshape(cam.width * cam.height, dSteps)
            renderedPoints = renderedPoints_Flat @ dAlpha

            gtImage = (torch.tensor(cv2.imread(cam.image,cv2.IMREAD_GRAYSCALE), dtype=torch.float64, device="cuda")/255.).T
            gtImage_Flat = gtImage.flatten()

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
                output_img = renderedPoints.reshape(cam.width, cam.height)
                torch.cuda.synchronize()
                showImg = output_img.T.cpu().detach().numpy()
                showGt = gtImage.T.cpu().detach().numpy()
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

    p30 = PerlinNoise3D(scale=2, res=30, center=testCenter, device="cuda")
    p10 = PerlinNoise3D(scale=2, res=10, center=testCenter, device="cuda")
    p3 = PerlinNoise3D(scale=2, res=3, center=testCenter, device="cuda")

    loss = train([p30,p10,p3], cams, 100, 0.01, False, False)
    torch.cuda.synchronize()
    plt.plot(loss)
    plt.show()