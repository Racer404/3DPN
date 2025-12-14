from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn

def lowmem_exact_spiral_index(requestCorners: torch.Tensor):
    device = requestCorners.device
    N = requestCorners.size(0)

    x = requestCorners[:, 0]
    y = requestCorners[:, 1]
    z = requestCorners[:, 2]

    # Build the 6 vectors without stacking a large [N, 6]
    vals = torch.stack([
        x,          # 0
        y,          # 1
        z,          # 2
        -x,         # 3
        -y,         # 4
        -z          # 5
    ], dim=1)       # [N, 6]

    # This tensor is only 6*N ints → for 20M: 20M*6*4 bytes = 480 MB int32, or 960 MB int64
    # To keep memory low, ensure int32 here:
    vals = vals.to(torch.int32)

    # EXACT face index (same tie-breaking as original)
    nLayer, face_idx = vals.max(dim=1)

    # Convert nLayer from signed value to positive magnitude
    nLayer = nLayer.abs()

    # Compute d, d1, d2
    d  = 2 * nLayer + 1
    d1 = d - 1
    d2 = d - 2

    # Core cube count
    core_count = (2*nLayer - 1).clamp(min=0).pow(3)

    # PREVIOUS layer counts (no 6×N stack)
    s0 = d  * d
    s1 = d  * d1
    s2 = d1 * d1
    s3 = s2
    s4 = d1 * d2

    c0 = s0
    c1 = c0 + s1
    c2 = c1 + s2
    c3 = c2 + s3
    c4 = c3 + s4

    prev = torch.zeros_like(c0)
    prev = torch.where(face_idx == 1, c0, prev)
    prev = torch.where(face_idx == 2, c1, prev)
    prev = torch.where(face_idx == 3, c2, prev)
    prev = torch.where(face_idx == 4, c3, prev)
    prev = torch.where(face_idx == 5, c4, prev)

    # real axis index
    real_idx = torch.remainder(face_idx, 3)  # 0,1,2 → x,y,z

    # Build u,v identical to original mask-based ordering
    cond0 = real_idx == 0
    cond1 = real_idx == 1

    u = torch.where(cond0, y, x)
    v = torch.where(real_idx == 2, y, z)

    # Shift
    u = u + nLayer
    v = v + nLayer

    # Compute layer_count same as original
    l0 = d  * u + v
    l1 = d  * u + v
    l2 = d1 * u + v
    l3 = d1 * u + v
    l4 = d1 * (u - 1) + v
    l5 = d2 * (u - 1) + (v - 1)

    layer_count = torch.zeros_like(l0)
    layer_count = torch.where(face_idx == 0, l0, layer_count)
    layer_count = torch.where(face_idx == 1, l1, layer_count)
    layer_count = torch.where(face_idx == 2, l2, layer_count)
    layer_count = torch.where(face_idx == 3, l3, layer_count)
    layer_count = torch.where(face_idx == 4, l4, layer_count)
    layer_count = torch.where(face_idx == 5, l5, layer_count)

    return core_count + prev + layer_count

def optimized_index3DSpiralByCorner(requestCorners: torch.Tensor):
    device = requestCorners.device
    N = requestCorners.size(0)

    req_conj = torch.cat([requestCorners, -requestCorners], dim=1)
    nLayer, face_idx = req_conj.max(dim=1)

    core_count = (2 * nLayer - 1).pow(3).clamp_min_(0)

    d  = 2 * nLayer + 1
    d1 = d - 1
    d2 = d - 2

    # face sizes
    s0 = d  * d
    s1 = d  * d1
    s2 = d1 * d1
    s3 = s2
    s4 = d1 * d2

    c0 = s0
    c1 = c0 + s1
    c2 = c1 + s2
    c3 = c2 + s3
    c4 = c3 + s4

    previous_layer_count = torch.stack([
        torch.zeros_like(c0),  # face0
        c0, c1, c2, c3, c4     # face1~5
    ], dim=1).gather(1, face_idx.unsqueeze(1)).squeeze(1)

    # two coords for face
    conj_idx = face_idx - 3
    real_idx = torch.where(conj_idx < 0, face_idx, conj_idx)

    idx = torch.arange(3, device=device).repeat(N, 1)
    mask = idx != real_idx.unsqueeze(1)
    layer_coord = requestCorners[mask].view(N, 2)

    u = layer_coord[:, 0] + nLayer
    v = layer_coord[:, 1] + nLayer

    l0 = d  * u + v
    l1 = d  * u + v
    l2 = d1 * u + v
    l3 = d1 * u + v
    l4 = d1 * (u - 1) + v
    l5 = d2 * (u - 1) + (v - 1)

    layer_count = torch.stack([l0, l1, l2, l3, l4, l5], dim=1) \
        .gather(1, face_idx.unsqueeze(1)).squeeze(1)

    return core_count + previous_layer_count + layer_count

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
    data = torch.load(path,weights_only=True)
    res = data['res']
    center = data['center']
    channelNum = data['channelNum']
    device = data['device']
    cornerVecs = data['cornerVecs']

    perlin = PerlinNoise3D(res, center, channelNum, device)
    perlin.cornerVecs = cornerVecs

    return perlin

class PerlinNoise3D(nn.Module):
    def __init__(self,
                 res:int = None,
                 center = torch.tensor([0.,0.,0.],dtype=torch.float32),
                 channelNum = None,
                 device:str = None
                 ):
        super().__init__()
        self.loss = None

        self.res = res
        self.center = center.to(device=device)
        self.channelNum = channelNum
        self.device = device
        initial_D = (torch.rand([(2 * self.res + 1) ** 3, 3, self.channelNum], dtype=torch.float32, device=device) - 0.5) * 2.   #[-1, 1], offset vector also [-1, 1]
        self.cornerVecs = nn.Parameter(initial_D)

    def writeTensor(self, path):
        writingPath = path
        torch.save({'res': self.res, 'center': self.center, 'channelNum':self.channelNum, 'device':self.device,'cornerVecs': self.cornerVecs}, writingPath)
        pass

    def extendCorners(self, target_idx, optimizer):
        target_idx = target_idx + 1
        needed = target_idx - self.cornerVecs.shape[0]
        if needed <= 0:
            return
        if optimizer is not None:
            new_rows = torch.randn(needed, 3, self.channelNum, device=self.cornerVecs.device)
            new_param = nn.Parameter(torch.cat([self.cornerVecs.data, new_rows], dim=0))
            self.cornerVecs = new_param
            optimizer.param_groups[0]['params'] = [self.cornerVecs]
        else:
            new_rows = torch.zeros(needed, 3, self.channelNum, device=self.cornerVecs.device)
            new_param = nn.Parameter(torch.cat([self.cornerVecs.data, new_rows], dim=0))
            self.cornerVecs = new_param
        print(f"Extended parameter to size {target_idx}")

    def getValue(self, requestedPoints, opt):
        requestedPoints_ = requestedPoints - self.center
        total_points = requestedPoints.shape[0]
        chunk = 500_000  # Chunk to control the GPU usage per batch
        gradient_out = torch.empty((8, total_points, self.channelNum), device=self.device)

        for i in range(0, total_points, chunk):
            _requestedPoints_ = requestedPoints_[i:(i + chunk)]
            corners, offsets = getCornerByCoor(self.res, _requestedPoints_)
            corners_flat = corners.reshape([-1, 3])
            reqVec_idx = lowmem_exact_spiral_index(corners_flat)
            if reqVec_idx.max() >= self.cornerVecs.shape[0]:
                self.extendCorners(int(reqVec_idx.max().item()), opt)
            reqVec_idx = reqVec_idx.long()  # <-- fix
            offsets = torch.transpose(offsets, 0, 1)

            corner_chunk = self.cornerVecs.index_select(0, reqVec_idx).reshape(-1,8,3,self.channelNum).transpose(0,1)
            off_chunk = offsets.unsqueeze(dim=-1)
            grad_chunk = torch.sum(off_chunk * corner_chunk, dim=2)
            gradient_out[:,i:i+chunk] = grad_chunk

        coord_inGrid = getCoorInGrid(self.res, requestedPoints_)
        smthSteps = lerpFunction(coord_inGrid).unsqueeze(dim=-1)
        value = trilinearInt(smthSteps, gradient_out) #Distribution: where X~[-0.6,0.6], Y~[0,1]
        return value
