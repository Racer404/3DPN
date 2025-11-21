import torch

def returnIndex(requestCorners: torch.Tensor):
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
    print(f"core_count:{core_count}")
    print(f"previous_layer_count:{previous_layer_count}")
    print(f"layer_count:{layer_count}")
    print(f"finalIdx:{finalIdx}")
    pass

n = 5000
requestCorners = ((torch.rand(n,3)-0.5) * 10).int()

returnIndex(requestCorners)