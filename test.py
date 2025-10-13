import torch
import utils
import cv2
from learnablePerlin3D import PerlinNoise3D
from matplotlib import pyplot as plt


dataset = "kitchen"
cams = utils.readColmapSceneInfo(dataset)
cam = cams[0]
perlin = PerlinNoise3D(scale=1, res=20, device="cuda", center=torch.tensor([-0.461083, 1.5, 1.5], dtype=torch.float64))

dSteps = 100  # Ray sampling frequency
dAlpha = utils.smoothStepsFunc(dSteps).to(device=cam.device)

dClose, dFar = cam.getDepthRange(perlin)
samplePoints_Volume, validPoints = cam.sampleVolumeBySteps(dClose, dFar, dSteps)
renderedPoints_Volume, output_mask_Volume = perlin.getValue(samplePoints_Volume, validPoints)
renderedPoints_Volume[~output_mask_Volume] = 0.5 # Background
renderedPoints_Flat = renderedPoints_Volume.reshape(cam.width * cam.height, dSteps)
renderedPoints = renderedPoints_Flat @ dAlpha

output_mask_Flat = output_mask_Volume.reshape(cam.width * cam.height, dSteps)

output_mask = torch.any(output_mask_Flat, dim=1)
# renderedPoints[~output_mask] = 0.5  # Background

output = renderedPoints.reshape(cam.width, cam.height)
image = output.T.cpu().detach().numpy()

gt = cv2.imread(cam.image,cv2.IMREAD_GRAYSCALE)/255.

plt.imshow(image)
plt.imshow(gt)
plt.show()