from learnablePerlin3D import PerlinNoise3D
import utils
from plViewer import PerlinViewer
import torch

dataset = "kitchen"
cams = utils.readColmapSceneInfo(dataset)
points = utils.readColmapPoints(dataset)
viewerCam = cams[0]
dAlpha = utils.smoothStepsFunc(100).to(device=viewerCam.device)

testCenter = torch.tensor([-0.461083, 1.5, 1.5], dtype=torch.float64, device="cuda")
perlin = PerlinNoise3D(scale=1, res=3, center=testCenter, channelNum=3, device="cuda")

viewer = PerlinViewer(viewerCam,[perlin],points)
viewer.start_loop()
viewer.app.run()