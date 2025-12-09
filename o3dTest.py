import learnablePerlin3D
import utils
from plViewer import PerlinViewer

dataset = "kitchen"
trainingSetup = "test_res=60_dSteps=100_decay_mse.9+ssim.1"
perlinFolder = f"{dataset}/trained/{trainingSetup}"

cams = utils.readColmapSceneInfo(dataset)
points = utils.readColmapPoints(dataset)
viewerCam = cams[0]

# perlin = learnablePerlin3D.readTensor(f"{perlinFolder}/0.pth")
perlin = learnablePerlin3D.PerlinNoise3D(scale = 1, res = 3, channelNum=4, device="cuda")

viewer = PerlinViewer(viewerCam, [perlin], points)
viewer.start_loop()
viewer.app.run()