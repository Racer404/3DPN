import torch
from matplotlib import  pyplot as plt
import learnablePerlin3D

import utils

cams = utils.readColmapSceneInfo("kitchen")
refCam = cams[0]

R_ = torch.tensor([[1,0,0],
                   [0,1,0],
                   [0,0,1]], device="cuda", dtype=torch.float64)
t_ = torch.tensor([ 0, 0, 1], device='cuda:0', dtype=torch.float64)

refCam.R = R_
refCam.t = t_
refCam.height = 389
refCam.intrinsic = torch.tensor([[403.9477,   0.0000, 194.6875],
                                        [0.0000, 403.9477, 194.6875],
                                        [  0.0000,   0.0000,   1.0000]], device='cuda:0', dtype=torch.float64)

perlin = learnablePerlin3D.PerlinNoise3D(scale=1, res=3, channelNum=1, device="cuda")
center_ = torch.tensor([0.,0.,0.],dtype=torch.float64, device="cuda")

p_close, p_far = refCam.getDepthRange(center_, 1)
requestPoints_Volume = refCam.sampleRayFixDepth(1)[0]

volume_out, grad_out = perlin.getValue(requestPoints_Volume)

volume_img = volume_out.reshape(refCam.width,refCam.height)

grad_img = grad_out.reshape(8, refCam.width, refCam.height)

plt.imshow(volume_img.cpu())
plt.show()

plt.imshow(grad_img[0].cpu())
plt.show()

plt.imshow(grad_img[1].cpu())
plt.show()

plt.imshow(grad_img[2].cpu())
plt.show()

plt.imshow(grad_img[3].cpu())
plt.show()

plt.imshow(grad_img[4].cpu())
plt.show()

plt.imshow(grad_img[5].cpu())
plt.show()

plt.imshow(grad_img[6].cpu())
plt.show()

plt.imshow(grad_img[7].cpu())
plt.show()

breakpoint()