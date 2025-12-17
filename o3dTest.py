import math

import learnablePerlin3D
import utils
from plViewer import PerlinViewer
import torch

dataset = "treehill"
cams = utils.readColmapSceneInfo(dataset)
optimalZ = utils.getDOIfromCams(cams)
res = 9
nyquist_freq = math.ceil(2 * (res * optimalZ * 2))

points = utils.readColmapPoints(dataset)
viewerCam = cams[0]

perlin = learnablePerlin3D.readTensor(f"{dataset}/trained/INF_res={res}_dStep={nyquist_freq}_raypass_mae.8+ssim.2/weights/0.pth")
viewer = PerlinViewer(viewerCam, perlin, optimalZ, nyquist_freq, points)
viewer.start_loop()
viewer.app.run()