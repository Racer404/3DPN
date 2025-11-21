import learnablePerlin3D
import utils
from plViewer import PerlinViewer

dataset = "kitchen"
cams = utils.readColmapSceneInfo(dataset)
points = utils.readColmapPoints(dataset)
viewerCam = cams[0]
dAlpha = utils.smoothStepsFunc(100).to(device=viewerCam.device)

# testCenter = torch.tensor([-0.461083, 1.5, 1.5], dtype=torch.float64, device="cuda")
perlin1 = learnablePerlin3D.readTensor(f"{dataset}/trained/test_p60*p60alpha/0.pth")
perlin2 = learnablePerlin3D.readTensor(f"{dataset}/trained/test_p60*p60alpha/1.pth")
# perlin3 = learnablePerlin3D.readTensor(f"{dataset}/trained/ds_shuffle_mse0.9_ssim0.1/2.pth")

viewer = PerlinViewer(viewerCam, [perlin1,perlin2], points)
viewer.start_loop()
viewer.app.run()