from learnablePerlin3D import PerlinNoise3D
import utils
import torch
from matplotlib import pyplot as plt

dataset = "kitchen"
cams = utils.readColmapSceneInfo(dataset)
viewerCam = cams[0]

channels = 3

dAlpha = utils.smoothStepsFunc(100).to(device=viewerCam.device)

testCenter = torch.tensor([-0.461083, 1.5, 1.5], dtype=torch.float64, device="cuda")
perlin = PerlinNoise3D(scale=1, res=3, center=testCenter, channelNum=channels, device="cuda")

dClose, dFar = viewerCam.getDepthRange(perlin)
samplePoints_Volume, validPoints = viewerCam.sampleVolumeBySteps(dClose, dFar, 100)

renderedPoints_vol, output_mask_Volume = perlin.getValue(samplePoints_Volume, validPoints)

renderedPoints_Flat = renderedPoints_vol.reshape(viewerCam.width * viewerCam.height, 100, channels)
renderedPoints = torch.matmul(renderedPoints_Flat.transpose(1, 2), dAlpha)
output = renderedPoints.reshape(viewerCam.width, viewerCam.height, channels)

image = output.transpose(0,1).cpu().detach().numpy()

plt.imshow(image)
plt.show()