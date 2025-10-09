import matplotlib.pyplot as plt
import torch

import utils
from learnablePerlin3D import PerlinNoise3D

dataset = "kitchen"
cam = utils.readColmapSceneInfo(dataset)[0]
testCenter = torch.tensor([-0.461083, 1.5, 1.5],dtype=torch.float64, device="cuda")
perlin = PerlinNoise3D(scale=1, res=15, center=testCenter, device="cuda")

dClose, dFar = cam.getDepthRange(perlin)
# samplePoints, validPoints = cam.sampleVolumeRandDepth(dClose, dFar)
dSteps = 100 #Ray sampling frequency
samplePoints_Flat, validPoints = cam.sampleVolumeBySteps(dClose, dFar, dSteps)
renderedPoints_Flat, output_mask = perlin.getValue(samplePoints_Flat, validPoints)

renderedPoints_Flat[~output_mask] = 0.5 #Background

renderedPoints_V = renderedPoints_Flat.reshape(cam.width * cam.height, dSteps)
dAlpha = utils.smoothStepsFunc(dSteps).to(device=cam.device)
renderedPoints = renderedPoints_V@dAlpha

output_img = renderedPoints.reshape(cam.width, cam.height)
torch.cuda.synchronize()
showImg = output_img.T.cpu().detach().numpy()

plt.imshow(showImg)
plt.show()
