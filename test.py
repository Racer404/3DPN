from typing import Tuple

import torch
from torch import Tensor


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

