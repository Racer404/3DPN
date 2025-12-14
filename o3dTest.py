import learnablePerlin3D
import utils
from plViewer import PerlinViewer

dataset = "kitchen"
trainingSetup = "scale=2_res=128_dSteps=256_decay_bg=0.5_mae.8+ssim.2"
perlinFolder = f"{dataset}/trained/{trainingSetup}"

cams = utils.readColmapSceneInfo(dataset)
points = utils.readColmapPoints(dataset)
viewerCam = cams[0]

p128 = learnablePerlin3D.readTensor(f"{perlinFolder}/weights/0.pth")

viewer = PerlinViewer(viewerCam, [p128], points, dSteps=256)
viewer.start_loop()
viewer.app.run()