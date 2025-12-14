import learnablePerlin3D
import utils
from plViewer import PerlinViewer
import torch

dataset = "counter"
cams = utils.readColmapSceneInfo(dataset)
points = utils.readColmapPoints(dataset)
viewerCam = cams[0]

# sceneCenter_kitchen = torch.tensor([-0.461083, 1.5, 1.5], dtype=torch.float32, device="cuda")
# perlin = learnablePerlin3D.PerlinNoise3D(res=2, center=sceneCenter_kitchen, device="cuda")

perlin = learnablePerlin3D.readTensor("counter/trained/test_INF_res=4_dStep=80_mae_close=0.5/0.pth")
viewer = PerlinViewer(viewerCam, perlin, points)
viewer.start_loop()
viewer.app.run()