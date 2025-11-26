from typing import Tuple

import torch
from torch import Tensor


def index3DSpiralByCorner(requestCorners: torch.Tensor):
    req_Len = requestCorners.size(0)

    req_conj = torch.cat([requestCorners, -requestCorners], dim=1)

    nLayer, face_idx = req_conj.max(dim=1)
    core_count = torch.pow((2 * nLayer - 1), 3).clip(min = 0)

    layer_diameter = 2 * nLayer + 1

    #Diameter of each face
    d0 = torch.stack([layer_diameter, layer_diameter], dim = 1)             #+x = n^2
    d1 = torch.stack([layer_diameter, layer_diameter-1], dim = 1)           #+y = n * (n-1)
    d2 = torch.stack([layer_diameter-1, layer_diameter-1], dim = 1)         #+z = (n-1)^2
    d3 = torch.stack([layer_diameter-1, layer_diameter-1], dim = 1)         #-x = (n-1)^2
    d4 = torch.stack([layer_diameter-1, layer_diameter-2], dim = 1)         #-y = (n-1) * (n-2)
    d5 = torch.stack([layer_diameter-2, layer_diameter-2], dim = 1)         #-z = (n-2)^2

    # Size of each face
    s0 = d0.prod(dim=1)         #+x = n^2
    s1 = d1.prod(dim=1)         #+y = (n-1) * n
    s2 = d2.prod(dim=1)         #+z = (n-1)^2
    s3 = d3.prod(dim=1)         #-x = (n-1)^2
    s4 = d4.prod(dim=1)         #-y = (n-1) * (n-2
    s_base = torch.zeros(req_Len, device=requestCorners.device)
    s_stack = torch.stack([s_base, s0, s1, s2, s3, s4], dim=1)  # shape [N, 6]
    cumsum_layer_size = s_stack.cumsum(dim=1)  # shape [N, 5]

    previous_layer_count = cumsum_layer_size[torch.arange(req_Len), face_idx]

    conj_idx = face_idx - 3
    real_idx = torch.where(conj_idx < 0, face_idx, conj_idx)
    layer_Coor_mask = (torch.arange(3, device=requestCorners.device).expand(req_Len, 3) != real_idx.unsqueeze(1))
    layer_coord = requestCorners[layer_Coor_mask].view(req_Len, 2)

    layer_coord_real = layer_coord + nLayer.unsqueeze(1)
    # layer_count of each layer
    l0 = (d0[:, 0]) * layer_coord_real[:, 0] + layer_coord_real[:, 1]                           #+x = d * u + v
    l1 = (d1[:, 0]) * layer_coord_real[:, 0] + layer_coord_real[:, 1]                           #+y = d * u + v
    l2 = (d2[:, 0]) * layer_coord_real[:, 0] + layer_coord_real[:, 1]                           #+z = (d-1) * u + v
    l3 = (d3[:, 0]) * layer_coord_real[:, 0] + layer_coord_real[:, 1]                           #-x = (d-1) * u + v
    l4 = (d4[:, 0]) * (layer_coord_real[:, 0] - 1) + layer_coord_real[:, 1]                     #-y = (d-1) * (u-1) + v
    l5 = (d5[:, 0]) * (layer_coord_real[:, 0] - 1) + (layer_coord_real[:, 1] - 1)               #-z = (d-1) * (u-1) + (v-1)
    l_stack = torch.stack([l0, l1, l2, l3, l4, l5], dim=1)  # shape [N, 6]
    layer_count = l_stack[torch.arange(req_Len), face_idx]


    finalIdx = core_count + previous_layer_count + layer_count

    # print(f"core_count:{core_count}")
    # print(f"previous_layer_count:{previous_layer_count}")
    # print(f"layer_count:{layer_count}")
    return finalIdx

def getCoorInGrid(res:int, reqCoor: torch.Tensor) -> torch.Tensor:
    unit = 1/res
    coor_inGrid = (reqCoor%unit)/unit
    return coor_inGrid

def getCornerByCoor(res:int, reqCoor: torch.Tensor) -> Tuple[Tensor, Tensor]:
    coords = reqCoor * res

    x0 = coords[:, 0].floor()
    x1 = coords[:, 0].ceil()
    y0 = coords[:, 1].floor()
    y1 = coords[:, 1].ceil()
    z0 = coords[:, 2].floor()
    z1 = coords[:, 2].ceil()

    corner000 = torch.stack([x0, y0, z0], dim=1)
    corner100 = torch.stack([x1, y0, z0], dim=1)
    corner010 = torch.stack([x0, y1, z0], dim=1)
    corner110 = torch.stack([x1, y1, z0], dim=1)
    corner001 = torch.stack([x0, y0, z1], dim=1)
    corner101 = torch.stack([x1, y0, z1], dim=1)
    corner011 = torch.stack([x0, y1, z1], dim=1)
    corner111 = torch.stack([x1, y1, z1], dim=1)

    corners = torch.stack([
        corner000,
        corner100,
        corner010,
        corner110,
        corner001,
        corner101,
        corner011,
        corner111,
    ], dim=1)

    offsets = coords.unsqueeze(dim=1).expand(-1,8,-1) - corners

    return corners, offsets

def linearInt(steps:torch.Tensor, corners:torch.Tensor):
    return corners[0]*(1.-steps)+corners[1]*steps

def bilinearInt(steps:torch.Tensor, corners:torch.Tensor):
    c12 = corners[0:2]
    c34 = corners[2:4]
    G12 = linearInt(steps[:,0],c12)
    G34 = linearInt(steps[:,0],c34)

    return G12*(1.-steps[:,1])+G34*steps[:,1]

def trilinearInt(steps:torch.Tensor, corners:torch.Tensor):
    c1234 = corners[0:4]
    c5678 = corners[4:8]
    G1234 = bilinearInt(steps[:,0:2],c1234)
    G5678 = bilinearInt(steps[:,0:2],c5678)

    return G1234*(1.-steps[:,2])+G5678*steps[:,2]

def lerpFunction(x):
    newX = 6 * x**5 - 15 * x**4 + 10 * x**3 #1-1 function
    # base = 6 * x**5 - 15 * x**4 + 10 * x**3
    # newX = base ** 2
    return newX

def readTensor(path):
    pass
    # data = torch.load(path,weights_only=True)
    #
    # scale = data['scale']
    # res = data['res']
    # center = data['center']
    # device = data['device']
    # cornerVecs = data['cornerVecs']
    #
    # perlin = PerlinNoise3D(scale, res, center, device)
    # perlin.cornerVecs = cornerVecs
    #
    # return perlin

class PerlinNoise3D:
    def __init__(self,
                 res:int = None,
                 center = torch.tensor([0.,0.,0.],dtype=torch.float64),
                 device:str = None
                 ):
        self.loss = None

        self.center = center.to(device=device)
        self.res = res
        self.device = device
        self.cornerVecs = (torch.rand([(2 * self.res + 1) ** 3, 3], dtype=torch.float64, device=device) - 0.5) * 2.   #[-1, 1], offset vector also [-1, 1]
        self.cornerVecs.requires_grad = False

    def writeTensor(self, path):
        # writingPath = path
        # torch.save({'scale': self.scale, 'res': self.tileNumber, 'center': self.center, 'device':self.device,'cornerVecs': self.cornerVecs}, writingPath)
        pass

    def extendCorners(self, targetIdx):
        targetIdx = targetIdx + 1
        newCorners = (torch.rand([targetIdx - self.cornerVecs.shape[0], 3], dtype=torch.float64, device=self.device) - 0.5) * 2.
        self.cornerVecs = torch.cat([self.cornerVecs, newCorners])

    def getValue(self, requestedPoints):
        requestedPoints_ = requestedPoints-self.center

        corners, offsets = getCornerByCoor(self.res, requestedPoints_)
        corners_flat = corners.reshape([-1,3])
        reqVec_idx = index3DSpiralByCorner(corners_flat)

        if reqVec_idx.max() >= self.cornerVecs.shape[0]:
            self.extendCorners(int(reqVec_idx.max().item()))
            print(f"extended to {reqVec_idx.max()} ?")

        chunk = 500_000 * 8 #Chunk to control the GPU usage per batch
        gradient_out = []

        reqVec_idx = reqVec_idx.long()  # <-- fix
        offsets = torch.transpose(offsets, 0, 1)
        for i in range(0, reqVec_idx.numel(), chunk):
            idx = reqVec_idx[i:i + chunk]
            corner_chunk = self.cornerVecs[idx].reshape(-1,8,3).transpose(0,1)
            off_chunk = offsets[:, i//8:(i + chunk)//8, :]
            grad_chunk = torch.sum(off_chunk * corner_chunk, dim=-1)
            gradient_out.append(grad_chunk)

        gradientVecs = torch.cat(gradient_out, dim=1)

        coord_inGrid = getCoorInGrid(self.res, requestedPoints_)
        smthSteps = lerpFunction(coord_inGrid)
        value = trilinearInt(smthSteps, gradientVecs) #Distribution: where X~[-0.6,0.6], Y~[0,1]

        return value
