def logical_to_linear(x, y, z):
    """
    Map a 3D logical coordinate (x,y,z) to a linear index,
    where index 0 is at (0,0,0) and points expand outward
    in cubic shells (square spiral in 3D).
    """
    # Layer / shell
    r = max(abs(x), abs(y), abs(z))

    if r == 0:
        return 0  # center point

    # Number of points in all previous layers
    base = (2 * r - 1) ** 3  # points in cube of side (2r-1)

    # Generate all points in current layer in deterministic order
    # z-major order: loop z, then y, then x
    points_in_layer = []
    for zz in range(-r, r + 1):
        for yy in range(-r, r + 1):
            for xx in range(-r, r + 1):
                if max(abs(xx), abs(yy), abs(zz)) == r:
                    points_in_layer.append((xx, yy, zz))

    # Find the index of (x,y,z) in this layer
    local_index = points_in_layer.index((x, y, z))

    return base + local_index



print(f"index:{logical_to_linear(-2412,1,1)}")
