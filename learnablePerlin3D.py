import torch

def index3DSpiral(requestCorners: torch.Tensor):
    req_Len = requestCorners.size(0)

    req_conj = torch.cat([requestCorners, -requestCorners], dim=1)

    nLayer, face_idx = req_conj.max(dim=1)
    core_count = torch.pow((2 * nLayer - 1), 3)

    layer_diameter = 2 * nLayer + 1

    #Diameter of each layer
    d0 = torch.stack([layer_diameter, layer_diameter], dim = 1)             #+x = n^2
    d1 = torch.stack([layer_diameter-1, layer_diameter], dim = 1)           #+y = (n-1) * n
    d2 = torch.stack([layer_diameter-1, layer_diameter-1], dim = 1)         #+z = (n-1)^2
    d3 = torch.stack([layer_diameter-1, layer_diameter-1], dim = 1)         #-x = (n-1)^2
    d4 = torch.stack([layer_diameter-1, layer_diameter-2], dim = 1)         #-y = (n-1) * (n-2)
    d5 = torch.stack([layer_diameter-2, layer_diameter-2], dim = 1)         #-z = (n-2)^2

    # Size of each layer
    s0 = d0.prod(dim=1)         #+x = n^2
    s1 = d1.prod(dim=1)         #+y = (n-1) * n
    s2 = d2.prod(dim=1)         #+z = (n-1)^2
    s3 = d3.prod(dim=1)         #-x = (n-1)^2
    s4 = d4.prod(dim=1)         #-y = (n-1) * (n-2
    s_base = torch.zeros(req_Len)
    s_stack = torch.stack([s_base, s0, s1, s2, s3, s4], dim=1)  # shape [N, 6]
    cumsum_layer_size = s_stack.cumsum(dim=1)  # shape [N, 5]

    previous_layer_count = cumsum_layer_size[torch.arange(req_Len), face_idx]

    conj_idx = face_idx - 3
    real_idx = torch.where(conj_idx < 0, face_idx, conj_idx)
    layer_Coor_mask = (torch.arange(3).expand(req_Len, 3) != real_idx.unsqueeze(1))
    layer_coord = requestCorners[layer_Coor_mask].view(req_Len, 2)

    layer_coord_real = layer_coord + nLayer.unsqueeze(1)

    # layer_count of each layer
    l0 = (d0[:, 0]) * layer_coord_real[:, 0] + layer_coord_real[:, 1]                   #+x = d * u + v
    l1 = (d1[:, 0] - 1) * layer_coord_real[:, 0] + layer_coord_real[:, 1]               #+y = (d-1) * u + v
    l2 = (d2[:, 0] - 1) * layer_coord_real[:, 0] + layer_coord_real[:, 1]               #+z = (d-1) * u + v
    l3 = (d3[:, 0] - 1) * layer_coord_real[:, 0] + layer_coord_real[:, 1]               #-x = (d-1) * u + v
    l4 = (d4[:, 0] - 1) * layer_coord_real[:, 0] + layer_coord_real[:, 1]               #-y = (d-1) * u + v
    l5 = (d5[:, 0] - 1) * layer_coord_real[:, 0] + layer_coord_real[:, 1]               #-z = (d-1) * u + v
    l_stack = torch.stack([l0, l1, l2, l3, l4, l5], dim=1)  # shape [N, 6]
    layer_count = l_stack[torch.arange(req_Len), face_idx]


    finalIdx = core_count + previous_layer_count + layer_count
    return finalIdx

def indexCornerByTile(n: int, cnVectors: torch.Tensor) -> torch.Tensor:
    grid_size = n + 1  # number of corners along each axis
    corner_indices = torch.arange(grid_size**3).reshape(grid_size, grid_size, grid_size)
    corner000 = corner_indices[:-1, :-1, :-1].reshape(-1)
    corner001 = corner_indices[:-1, :-1, 1:].reshape(-1)
    corner010 = corner_indices[:-1, 1:, :-1].reshape(-1)
    corner011 = corner_indices[:-1, 1:, 1:].reshape(-1)
    corner100 = corner_indices[1:, :-1, :-1].reshape(-1)
    corner101 = corner_indices[1:, :-1, 1:].reshape(-1)
    corner110 = corner_indices[1:, 1:, :-1].reshape(-1)
    corner111 = corner_indices[1:, 1:, 1:].reshape(-1)
    indexed_corners = torch.stack([
        cnVectors[corner000],  # (N, C)
        cnVectors[corner001],
        cnVectors[corner010],
        cnVectors[corner011],
        cnVectors[corner100],
        cnVectors[corner101],
        cnVectors[corner110],
        cnVectors[corner111]
    ])
    return indexed_corners

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

def gradientProduct2Img(gradientTensor,tileNumber,tileSize,resolution):
    outputGradient = gradientTensor
    outputGradient = outputGradient.reshape(4, tileNumber, tileNumber, tileSize ** 2)
    outputGradient = outputGradient.reshape(4, tileNumber, tileNumber, tileSize, tileSize).permute(0, 1, 3, 2, 4)
    outputGradient = outputGradient.reshape(4, resolution, resolution)
    return outputGradient

def readTensor(path):
    data = torch.load(path,weights_only=True)

    scale = data['scale']
    res = data['res']
    center = data['center']
    device = data['device']
    cornerVecs = data['cornerVecs']

    perlin = PerlinNoise3D(scale, res, center, device)
    perlin.cornerVecs = cornerVecs

    return perlin

class PerlinNoise3D:
    def __init__(self,
                 scale:float = None,
                 res:int = None,
                 center = torch.tensor([0.,0.,0.],dtype=torch.float64),
                 device:str = None
                 ):
        self.loss = None

        self.scale = scale
        self.center = center.to(device=device)
        self.tileNumber = res
        self.device = device
        self.tileSize = scale/float(res)
        self.cornerVecs = (torch.rand([(self.tileNumber+1)**3,3], dtype=torch.float64, device=device)-0.5) * 2.   #[-1, 1], offset vector also [-1, 1]
        self.cornerVecs.requires_grad = False
        self.offsets = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.float64, device=device)
        self.corner_Flat = None

    def writeTensor(self, path):
        writingPath = path
        torch.save({'scale': self.scale, 'res': self.tileNumber, 'center': self.center, 'device':self.device,'cornerVecs': self.cornerVecs}, writingPath)

    def getValue(self, requestedPoints):
        self.corner_Flat = indexCornerByTile(self.tileNumber, self.cornerVecs)

        toPerlinCenter = torch.tensor([0.5, 0.5, 0.5]).to(dtype=torch.float64, device=self.device) * self.scale
        validPoints = (requestedPoints - self.center) + toPerlinCenter

        coord_inGrid = (validPoints%self.tileSize)/self.tileSize
        offsetVecs = coord_inGrid[:,None] - self.offsets
        offsetVecs = torch.transpose(offsetVecs,0,1)

        requestedTile = torch.floor(validPoints/self.tileSize)
        requestedTile_idx = (requestedTile[:,2]*(self.tileNumber**2)+requestedTile[:,1]*self.tileNumber+requestedTile[:,0]).to(torch.int)

        chunk = 500_000   #Chunk to control the GPU usage per batch
        gradient_out = []
        requestedTile_idx = requestedTile_idx.long()  # <-- fix
        for i in range(0, requestedTile_idx.numel(), chunk):
            idx = requestedTile_idx[i:i + chunk]
            idx_expand = idx.view(1, -1, 1).expand(8, -1, 3)
            corner_chunk = torch.take_along_dim(self.corner_Flat, idx_expand, dim=1)
            off_chunk = offsetVecs[:, i:i + chunk, :]
            grad_chunk = torch.sum(off_chunk * corner_chunk, dim=-1)
            gradient_out.append(grad_chunk)

        gradientVecs = torch.cat(gradient_out, dim=1)
        smthSteps = lerpFunction(coord_inGrid)
        value = trilinearInt(smthSteps, gradientVecs) #Distribution: where X~[-0.6,0.6], Y~[0,1]

        return value
