import torch

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

class PerlinNoise3D:
    def __init__(self,
                 scale:float = None,
                 res:int = None,
                 center = torch.tensor([0.,0.,0.],dtype=torch.float32),
                 channelNum = 1,
                 device:str = None
                 ):
        self.loss = None

        self.scale = scale
        self.center = center.to(device=device)
        self.channelNum = channelNum
        self.tileNumber = res
        self.device = device
        self.tileSize = scale/float(res)
        self.cornerVecs = (torch.rand([(self.tileNumber+1)**3,3,self.channelNum], dtype=torch.float32, device=device)-0.5) * 2.
        self.cornerVecs.requires_grad = False
        self.offsets = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.float32, device=device)
        self.corner_Flat = None

    def writeTensor(self, path):
        writingPath = path
        torch.save({'scale': self.scale, 'res': self.tileNumber, 'center': self.center, 'channelNum':self.channelNum, 'device':self.device,'cornerVecs': self.cornerVecs}, writingPath)

    def getValue(self, requestedPoints):
        self.corner_Flat = indexCornerByTile(self.tileNumber, self.cornerVecs)

        toPerlinCenter = torch.tensor([0.5, 0.5, 0.5]).to(dtype=torch.float32, device=self.device) * self.scale
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
            idx_expand = idx.view(1, -1, 1, 1).expand(8, -1, 3, self.channelNum)
            corner_chunk = torch.take_along_dim(self.corner_Flat, idx_expand, dim=1)
            off_chunk = offsetVecs[:, i:i + chunk, :].unsqueeze(dim=-1)
            grad_chunk = torch.sum(off_chunk * corner_chunk, dim=2)
            gradient_out.append(grad_chunk)

        gradientVecs = torch.cat(gradient_out, dim=1)

        smthSteps = lerpFunction(coord_inGrid).unsqueeze(dim=-1)

        value = trilinearInt(smthSteps, gradientVecs) / 2. + 0.5

        return value

def readTensor(path) -> PerlinNoise3D:
    data = torch.load(path,weights_only=True)

    scale = data['scale']
    res = data['res']
    center = data['center']
    channelNum = data['channelNum']
    device = data['device']
    cornerVecs = data['cornerVecs']

    perlin = PerlinNoise3D(scale, res, center, channelNum, device)
    perlin.cornerVecs = cornerVecs

    return perlin