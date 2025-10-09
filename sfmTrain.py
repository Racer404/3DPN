import os

import cv2
import torch
from PIL import Image
from torch import optim

import utils
from learnablePerlin3D import PerlinNoise3D

def train(
        perlin: PerlinNoise3D = None,
        cameras: [utils.Camera] = None,
        iterations: int = None,
        lr: float = None,
        ifVisualize: bool = False,
        ifSaveResult: bool = False) -> torch.Tensor:
    optimizer = optim.Adam(
        [perlin.cornerVecs], lr
    )
    mse_loss = torch.nn.MSELoss()
    mae_loss = torch.nn.L1Loss()

    output_img = None
    frames = []
    perlin.loss = []
    dSteps = 100
    dAlpha = utils.smoothStepsFunc(dSteps).to(device=cams[0].device)



    for iter in range(iterations):
        for cam in cameras:
            dClose, dFar = cam.getDepthRange(perlin)
            samplePoints_Volume, validPoints = cam.sampleVolumeBySteps(dClose, dFar, dSteps)
            renderedPoints_Volume, output_mask_Volume = perlin.getValue(samplePoints_Volume, validPoints)
            renderedPoints_Volume[~output_mask_Volume] = 0.5

            renderedPoints_Flat = renderedPoints_Volume.reshape(cam.width * cam.height, dSteps)
            renderedPoints = renderedPoints_Flat @ dAlpha

            output_mask_Flat = output_mask_Volume.reshape(cam.width * cam.height, dSteps)
            output_mask = torch.any(output_mask_Flat, dim=1)

            gtImage = (torch.tensor(cv2.imread(cam.image,cv2.IMREAD_GRAYSCALE), dtype=torch.float64, device="cuda")/255.).T
            gtImage_Flat = gtImage.flatten()
            loss = mse_loss(renderedPoints[output_mask], gtImage_Flat[output_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")
            perlin.loss.append(loss.item())

            if ifVisualize:
                # renderedPoints[~output_mask] = 0.5 #Background
                output_img = renderedPoints.reshape(cam.width, cam.height)
                torch.cuda.synchronize()
                showImg = output_img.T.cpu().detach().numpy()
                showGt = gtImage.T.cpu().detach().numpy()
                cv2.imshow("Training", showImg)
                cv2.imshow("GT", showGt)
                frames.append(showImg)
                cv2.waitKey(1)

    if ifSaveResult:
        frames = [Image.fromarray(frame) for frame in frames]
        out_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(out_dir, exist_ok=True)
        frames[0].save(
            f"{out_dir}/training.gif",
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=5,
            loop=0,
        )

    return output_img


if __name__ == "__main__":

    dataset = "kitchen"

    cams = utils.readColmapSceneInfo(dataset)
    testCenter = torch.tensor([-0.461083, 1.5, 1.5],dtype=torch.float64, device="cuda")
    perlin = PerlinNoise3D(scale=1, res=15, center=testCenter, device="cuda")

    train(perlin, cams, 1000, 0.01, True, False)