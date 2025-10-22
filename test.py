from learnablePerlin3D import PerlinNoise3D
import learnablePerlin3D
import torch

perlin_A = PerlinNoise3D(1,3, device="cuda")
path = "kitchen/trained.pth"
perlin_A.writeTensor(path)

perlin_B = learnablePerlin3D.readTensor(path)
breakpoint()