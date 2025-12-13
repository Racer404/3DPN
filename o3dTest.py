import learnablePerlin3D
import utils
from plViewer import PerlinViewer

dataset = "kitchen"
trainingSetup = "scale=2_res=3+19+47_dSteps=100_decay_mae.8+ssim.2_bg=0.5_float32"
perlinFolder = f"{dataset}/trained/{trainingSetup}"

cams = utils.readColmapSceneInfo(dataset)
points = utils.readColmapPoints(dataset)
viewerCam = cams[0]

p3 = learnablePerlin3D.readTensor(f"{perlinFolder}/0.pth")
p20 = learnablePerlin3D.readTensor(f"{perlinFolder}/1.pth")
p50 = learnablePerlin3D.readTensor(f"{perlinFolder}/2.pth")

viewer = PerlinViewer(viewerCam, [p3, p20, p50], points)
viewer.start_loop()
viewer.app.run()