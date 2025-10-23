import learnablePerlin3D
import utils
from plViewer import PerlinViewer

dataset = "kitchen"
cams = utils.readColmapSceneInfo(dataset)
points = utils.readColmapPoints(dataset)
viewerCam = cams[0]

perlin1 = learnablePerlin3D.readTensor("kitchen/trained/0.pth")
perlin2 = learnablePerlin3D.readTensor("kitchen/trained/1.pth")
perlin3 = learnablePerlin3D.readTensor("kitchen/trained/2.pth")

perlin_old = learnablePerlin3D.readTensor("kitchen/trained.pth")

viewer = PerlinViewer(viewerCam, [perlin_old], points)
viewer.start_loop()
viewer.app.run()