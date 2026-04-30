import math

import learnablePerlin3D
import utils
from o3dRenderer import PerlinViewer

dataset = "room"
cams = utils.readColmapSceneInfo(f"data/{dataset}")
optimalZ = utils.getDOIfromCams(cams)
res = 7
nyquist_freq = math.ceil(2 * (res * optimalZ * 2))

points = utils.readColmapPoints(f"data/{dataset}")
viewerCam = cams[0]

perlin = learnablePerlin3D.readTensor(f"data/{dataset}/trained/INF_res={res}_dStep={nyquist_freq}_raypass_mae.8+ssim.2/weights/0.pth")
viewer = PerlinViewer(viewerCam, perlin, optimalZ, nyquist_freq, points)
viewer.start_loop()
viewer.app.run()