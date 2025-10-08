import os

import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import optim

import utils
from learnablePerlin3D import PerlinNoise3D
from matplotlib import pyplot as plot

dataset = "kitchen"
cam = utils.readColmapSceneInfo(dataset)[0]
testCenter = torch.tensor([-0.461083, 1.5, 1.5],dtype=torch.float64, device="cuda")
perlin = PerlinNoise3D(scale=1, res=15, center=testCenter, device="cuda")

dClose, dFar = cam.getDepthRange(perlin)
# samplePoints, validPoints = cam.sampleVolumeRandDepth(dClose, dFar)
samplePoints, validPoints = cam.sampleVolumeBySteps(dClose, dFar, 10)
renderedPoints, output_mask = perlin.getValue(samplePoints, validPoints)

renderedPoints[~output_mask] = 0.5 #Background
output_img = renderedPoints.reshape(389, 260, 1).squeeze()
torch.cuda.synchronize()
showImg = output_img.T.cpu().detach().numpy()

plt.imshow(showImg)
plt.show()
