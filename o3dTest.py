import learnablePerlin3D
import utils
from plViewer import PerlinViewer

dataset = "kitchen"
cams = utils.readColmapSceneInfo(dataset)
points = utils.readColmapPoints(dataset)
viewerCam = cams[0]

perlin_sigmoid = learnablePerlin3D.readTensor("LNPL Data analysis/sige^1autoScale(x)_shuttle_mse/0.pth")

viewer = PerlinViewer(viewerCam, [perlin_sigmoid], points)
viewer.start_loop()
viewer.app.run()