import learnablePerlin3D
import utils
from plViewer import PerlinViewer

dataset = "kitchen"
cams = utils.readColmapSceneInfo(dataset)
points = utils.readColmapPoints(dataset)
viewerCam = cams[0]

perlin = learnablePerlin3D.readTensor("kitchen/trained.pth")

viewer = PerlinViewer(viewerCam, perlin, points)
viewer.start_loop()
viewer.app.run()