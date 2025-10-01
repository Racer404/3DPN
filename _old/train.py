import torch
from _old import learnablePerlin
from matplotlib import pyplot as plt
import cv2

# input = torch.rand(20,20,device="cuda",dtype=torch.float32)*2-1
# input_tensor = input.unsqueeze(0).unsqueeze(0)
# m = torch.nn.Upsample(scale_factor=20, mode='nearest')
# groundTruth = m(input_tensor).squeeze(0).squeeze(0)
imgFile = cv2.imread("../mountain.png", cv2.IMREAD_GRAYSCALE)
groundTruth = (torch.tensor(imgFile,device="cuda",dtype=torch.float32)-127.5)/127.5

perlin = learnablePerlin.PerlinNoise2D(640, 64, "cuda")

plt.imshow(perlin.render().cpu().detach().numpy())

plt.show()
