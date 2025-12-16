import math

import learnablePerlin3D
import utils
from plViewer import PerlinViewer
import torch

dataset = "room"
cams = utils.readColmapSceneInfo(dataset)
optimalZ = utils.getDOIfromCams(cams)
res = 7
nyquist_freq = math.ceil(2 * (res * optimalZ * 2))

points = utils.readColmapPoints(dataset)
viewerCam = cams[0]

perlin = learnablePerlin3D.readTensor("room/trained/INF_res=7_dStep=113_raypass_mae.8+ssim.2/weights/0.pth")
viewer = PerlinViewer(viewerCam, perlin, optimalZ, nyquist_freq, points)
viewer.start_loop()
viewer.app.run()