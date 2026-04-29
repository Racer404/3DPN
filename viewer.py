import learnablePerlin3D
import utils
from o3dRenderer import PerlinViewer

dataset = "kitchen"
trainingSetup = "scale=4.25_res=64+16+4_dSteps=128_decay_bg=0.5_mae.8+ssim.2"
perlinFolder = f"data/{dataset}/trained/{trainingSetup}"

cams = utils.readColmapSceneInfo(f"data/{dataset}")
points = utils.readColmapPoints(f"data/{dataset}")
viewerCam = cams[0]

p64 = learnablePerlin3D.readTensor(f"{perlinFolder}/weights/0.pth")
p16 = learnablePerlin3D.readTensor(f"{perlinFolder}/weights/1.pth")
p4 = learnablePerlin3D.readTensor(f"{perlinFolder}/weights/2.pth")

viewer = PerlinViewer(viewerCam, [p64, p16, p4], points, dSteps=128)
viewer.start_loop()
viewer.app.run()