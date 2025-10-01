import torch
from torch import optim
import cv2
from PIL import Image
import numpy
import os

def get_corner_group(n: int, cnVectors: torch.Tensor) -> torch.Tensor:
    grid_size = n + 1  # number of corners along each axis
    corner_indices = torch.arange(grid_size**2).reshape(grid_size, grid_size)
    return torch.stack([cnVectors[corner_indices[:-1, :-1].reshape(-1)],cnVectors[corner_indices[:-1, 1:].reshape(-1)],cnVectors[corner_indices[1:, :-1].reshape(-1)],cnVectors[corner_indices[1:, 1:].reshape(-1)]])


def lerpFunction(x):
    newX = 6 * x**5 - 15 * x**4 + 10 * x**3
    # base = 6 * x**5 - 15 * x**4 + 10 * x**3
    # newX = base ** 2
    return newX

def gradientProduct2Img(gradientTensor,tileNumber,tileSize,resolution):
    outputGradient = gradientTensor
    outputGradient = outputGradient.reshape(4, tileNumber, tileNumber, tileSize ** 2)
    outputGradient = outputGradient.reshape(4, tileNumber, tileNumber, tileSize, tileSize).permute(0, 1, 3, 2, 4)
    outputGradient = outputGradient.reshape(4, resolution, resolution)
    return outputGradient

class PerlinNoise2D:
    def __init__(self,
                 res:int = 0,
                 tileSize:int = 0,
                 device:str = ""
                 ):
        self.loss = None

        self.interpolated_F = None
        self.interpolated_UP = None
        self.interpolated_BL = None
        self.gradientMap = None
        self.tileSize = tileSize
        self.res = res
        self.device = device
        self.tileNumber = int(res/tileSize)
        self.cornerVecs = (torch.rand([(self.tileNumber+1)**2,2], dtype=torch.float32)-0.5)*2.
        # self.cornerVecs = (torch.ones([(self.tileNumber+1)**2,2], dtype=torch.float32)) #DEBUG ONLY
        self.offsetMap = ((torch.stack(torch.meshgrid([torch.arange(0,tileSize),torch.arange(0,tileSize)],indexing='ij'),dim=-1).to(torch.float32)+0.5)/tileSize).tile((4,1,1,1))
        offsets = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.float32)
        offsets = offsets[:, None, None, :]  # reshape to (4, 1, 1, 2)
        self.offsetMap -= offsets
        self.offsetMap = self.offsetMap.transpose(1,2)
        self.offsetMap = self.offsetMap.reshape(4,tileSize**2,2).transpose(1,2)

        step = torch.arange(0,self.tileSize,dtype=torch.float32)/self.tileSize
        self.lerpMatrix = (torch.tile(lerpFunction(step),[self.tileNumber]))
        self.lerpMatrix = self.lerpMatrix.to(device=device)
        self.cornerVecs = self.cornerVecs.to(device=device)
        self.offsetMap = self.offsetMap.to(device=device)
        self.cornerVecs.requires_grad = True

    def render(self) -> torch.Tensor:
        gradientMap = (get_corner_group(self.tileNumber,self.cornerVecs)@self.offsetMap)
        gradientMap = gradientProduct2Img(gradientMap,self.tileNumber,self.tileSize,self.res)
        self.gradientMap = gradientMap
        self.interpolated_UP = self.gradientMap[0] * (1.-self.lerpMatrix) + self.gradientMap[1] * self.lerpMatrix
        self.interpolated_BL = self.gradientMap[2] * (1.-self.lerpMatrix) + self.gradientMap[3] * self.lerpMatrix
        self.interpolated_F = self.interpolated_UP.t() * (1.-self.lerpMatrix) + self.interpolated_BL.t() * self.lerpMatrix
        return self.interpolated_F

    def train(self,
        iterations: int = None,
        lr: float = None,
        gtImage: torch.Tensor = None,
        ifVisualize: bool = False,
        ifSaveResult: bool = False)->torch.Tensor:
        optimizer = optim.Adam(
            [self.cornerVecs], lr
        )
        mse_loss = torch.nn.MSELoss()
        mae_loss = torch.nn.L1Loss()
        rendered = None
        frames = []
        self.loss = []
        for iter in range(iterations):
            rendered = self.render()
            loss = mse_loss(rendered,gtImage)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")
            self.loss.append(loss.item())
            if ifVisualize:
                torch.cuda.synchronize()
                showImg = cv2.normalize(rendered.cpu().detach().numpy(),None,0,255,norm_type=cv2.NORM_MINMAX).astype(numpy.uint8)
                cv2.imshow("Training in process", showImg)
                frames.append(showImg)
                cv2.waitKey(1)
        if ifSaveResult:
            frames = [Image.fromarray(frame) for frame in frames]
            out_dir = os.path.join(os.getcwd(), "results")
            os.makedirs(out_dir, exist_ok=True)
            frames[0].save(
                f"{out_dir}/training.gif",
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )
        return rendered