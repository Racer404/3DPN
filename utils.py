import os
import numpy
import cv2
import torch
from typing import List

from colmap_loader import read_extrinsics_binary, read_intrinsics_binary, read_extrinsics_text, read_intrinsics_text, read_points3D_binary, read_points3D_text

def readColmapSceneInfo(path):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    n = len(cam_extrinsics)
    raw_h = cv2.imread(path + "/images/" + cam_extrinsics[0].name).shape[1]
    downsample = raw_h / 400.
    images_path = [path + "/images_400/" + cam_extrinsics[i].name for i in range(n)]
    raw = cv2.imread(images_path[0])
    h,w = raw.shape[0:2]
    intrinsic = torch.tensor(
        [[cam_intrinsics[1].params[0]/downsample,0,cam_intrinsics[1].params[2]/downsample],
         [0,cam_intrinsics[1].params[1]/downsample,cam_intrinsics[1].params[3]/downsample],
         [0,0,1]]).to(dtype=torch.float32)
    poses_R = torch.tensor(numpy.array([cam_extrinsics[i].qvec2rotmat() for i in range(n)])).to(dtype=torch.float32)
    poses_t = torch.tensor(numpy.array([cam_extrinsics[i].tvec for i in range(n)])).to(dtype=torch.float32)

    cameras = [Camera(width=w,height=h,intrinsic=intrinsic,R=poses_R[i],t=poses_t[i],image=images_path[i]) for i in range(n)]
    return cameras


def smoothStepsFunc(steps):
    steps_n = torch.arange(steps, dtype=torch.float32) / steps
    steps_n = torch.cat([steps_n[1:], torch.tensor([1])])
    func = -(steps_n ** 2) + 2 * steps_n  # Looking for a function from [0,1] -> [0,1]
    alpha = func - torch.cat([torch.tensor([0]), func[:-1]])
    return alpha

def renderVolume_stepsMean(color_Valid:torch.Tensor, alpha_Valid:torch.Tensor, volume_Mask:torch.Tensor, dSteps: int):
    volume_Valid = (color_Valid * alpha_Valid).mean(dim=0)
    channels = volume_Valid.shape[1]
    volume = volume_Mask.shape[0]
    renderedPoints_Volume = torch.zeros([volume, channels], dtype=torch.float32, device=volume_Valid.device)
    renderedPoints_Volume[volume_Mask] = volume_Valid
    renderedPoints_Volume[~volume_Mask] = 0.5

    renderedPoints_dLayers = renderedPoints_Volume.reshape(-1, dSteps, channels)
    mask_dLayers = volume_Mask.reshape(-1, dSteps)

    rendered_Flat = renderedPoints_dLayers.mean(dim=1)    #Every points on a ray
    mask_Flat = torch.any(mask_dLayers, dim=1)

    return rendered_Flat, mask_Flat

def renderVolume_stepsMeanValid(color_Valid:torch.Tensor, alpha_Valid:torch.Tensor, volume_Mask:torch.Tensor, dSteps: int):
    volume_Valid = (color_Valid * alpha_Valid).mean(dim=0)
    channels = volume_Valid.shape[1]
    volume = volume_Mask.shape[0]
    renderedPoints_Volume = torch.zeros([volume, channels], dtype=torch.float32, device=volume_Valid.device)
    renderedPoints_Volume[volume_Mask] = volume_Valid

    renderedPoints_dLayers = renderedPoints_Volume.reshape(-1, dSteps, channels)
    mask_dLayers = volume_Mask.reshape(-1, dSteps)

    rendered_Flat = renderedPoints_dLayers.sum(dim=1)/mask_dLayers.sum(dim=1).unsqueeze(-1)     #Only consider valid points on a ray
    mask_Flat = torch.any(mask_dLayers, dim=1)

    return rendered_Flat, mask_Flat

def renderVolume_stepsDecay(color_Valid:torch.Tensor, alpha_Valid:torch.Tensor, volume_Mask:torch.Tensor, dSteps: int):
    volume_Valid = (color_Valid * alpha_Valid).mean(dim=0)
    channels = volume_Valid.shape[1]
    volume = volume_Mask.shape[0]
    renderedPoints_Volume = torch.zeros([volume, channels], dtype=torch.float32, device=volume_Valid.device)
    renderedPoints_Volume[volume_Mask] = volume_Valid
    renderedPoints_Volume[~volume_Mask] = 0.5

    renderedPoints_dLayers = renderedPoints_Volume.reshape(-1, dSteps, channels)
    mask_dLayers = volume_Mask.reshape(-1, dSteps)

    dAlpha = smoothStepsFunc(dSteps).to(device=volume_Valid.device)
    rendered_Flat = torch.matmul(renderedPoints_dLayers.transpose(1, 2), dAlpha)

    mask_Flat = torch.any(mask_dLayers, dim=1)
    return rendered_Flat, mask_Flat

def renderVolume_stepsRaypass(color_Valid:torch.Tensor, alpha_Valid:torch.Tensor, volume_Mask:torch.Tensor, dSteps: int):
    color_Valid = color_Valid.mean(dim=0)
    alpha_Valid = alpha_Valid.mean(dim=0)

    volume = volume_Mask.shape[0]
    channels = color_Valid.shape[1]

    renderedColor_Volume = torch.zeros([volume, channels], dtype=torch.float32, device=volume_Mask.device)
    renderedColor_Volume[volume_Mask] = color_Valid
    renderedColor_Volume[~volume_Mask] = 0.
    renderedColor_dLayers = renderedColor_Volume.reshape(-1, dSteps, channels)

    renderedAlpha_Volume = torch.zeros([volume, 1], dtype=torch.float32, device=volume_Mask.device)
    renderedAlpha_Volume[volume_Mask] = alpha_Valid
    renderedAlpha_Volume[~volume_Mask] = 0.
    renderedAlpha_dLayers = renderedAlpha_Volume.reshape(-1, dSteps, 1)

    delta = 0.01
    alpha_norm = 1.0 - torch.exp(-renderedAlpha_dLayers * delta)

    # Transmittance
    _alpha = 1.0 - alpha_norm
    T = torch.cumprod(
        torch.cat([torch.ones_like(_alpha[:, :1]), _alpha[:, :-1]], dim=1),
        dim=1
    )  # [N, Z]

    weights = T * alpha_norm # [N, Z, 1]

    rendered_Flat = torch.sum(renderedColor_dLayers * weights, dim=1)

    mask_dLayers = volume_Mask.reshape(-1, dSteps)
    mask_Flat = torch.any(mask_dLayers, dim=1)
    return rendered_Flat, mask_Flat

def maskValidPoints(requestedPoints, p_center, p_scale):
    toPerlinCenter = torch.tensor([0.5, 0.5, 0.5]).to(dtype=torch.float32, device="cuda") * p_scale
    requestedPoints = (requestedPoints - p_center) + toPerlinCenter

    mask = (requestedPoints >= 0) & (requestedPoints < p_scale)
    valid_mask = mask.all(dim=1)

    return valid_mask


class Camera:
    def __init__(self, width, height, intrinsic, R, t, image):
        self.device = "cuda"
        self.width = width
        self.height = height
        self.intrinsic = intrinsic.to(device=self.device)
        self.R = R.to(device=self.device)
        self.t = t.to(device=self.device)
        self.image = image

    def Rt2Mat4(self):
        lastRow = torch.tensor([0,0,0,1], dtype=torch.float32, device=self.device)
        upperPart = torch.concatenate([self.R, self.t.unsqueeze(1)], dim=1)
        mat4_4_Rt = torch.concatenate([upperPart, lastRow.unsqueeze(0)])
        return mat4_4_Rt


    def getDepthRange(self, center, scale):
        cornerCoords = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.float32, device=self.device)
        toPerlinCenter = torch.tensor([0.5, 0.5, 0.5]).to(dtype=torch.float32, device=self.device)

        cornerCoords = scale * (cornerCoords - toPerlinCenter) + center
        cornerCoords_Camera = (self.R @ cornerCoords.T).T + self.t
        cornerDepth_Camera =  cornerCoords_Camera[:,2]

        return cornerDepth_Camera.min().item(), cornerDepth_Camera.max().item()

    def sampleVolumeBySteps(self, dClose:float, dFar:float, steps: int):
        coords_inImage = torch.stack(torch.meshgrid([torch.arange(0,self.width),torch.arange(0,self.height)],indexing='ij'),dim=-1)
        coords_inImage_Flat = coords_inImage.reshape([-1,2])
        coords_inImage_Flat_Homo = torch.hstack([coords_inImage_Flat,torch.ones(coords_inImage_Flat.shape[0]).unsqueeze(dim=-1)]).to(dtype=torch.float32,device=self.device)
        coords_inCamera_Flat_Homo = (torch.linalg.inv(self.intrinsic)@coords_inImage_Flat_Homo.T).T

        steps_norm = torch.arange(0, 1, 1 / steps, dtype=torch.float32)
        steps_norm = torch.cat([steps_norm[1:], torch.tensor([1.])]).to(device=self.device)
        steps_depth = steps_norm * (dFar-dClose) + dClose

        samplePoints = coords_inCamera_Flat_Homo.unsqueeze(-1) @ steps_depth.unsqueeze(-1).T
        samplePoints = samplePoints.transpose(1,2) # [N, steps, 3]
        samplePoints_Flat = samplePoints.reshape([-1,3]) # [N*steps, 3]

        validIdx = samplePoints_Flat[:, 2] > 0
        samplePoints_World_Flat = (self.R.T @ (samplePoints_Flat - self.t).T).T

        return samplePoints_World_Flat, validIdx

    def sampleRayRandDepth(self, dClose:float, dFar:float):
        coords_inImage = torch.stack(torch.meshgrid([torch.arange(0,self.width),torch.arange(0,self.height)],indexing='ij'),dim=-1)
        coords_inImage_Flat = coords_inImage.reshape([-1,2])
        coords_inImage_Flat_Homo = torch.hstack([coords_inImage_Flat,torch.ones(coords_inImage_Flat.shape[0]).unsqueeze(dim=-1)]).to(dtype=torch.float32,device=self.device)
        coords_inCamera_Flat_Homo = (torch.linalg.inv(self.intrinsic)@coords_inImage_Flat_Homo.T).T

        randomDepth = (torch.rand(coords_inCamera_Flat_Homo.shape[0],device=self.device)*(dFar-dClose)+dClose).unsqueeze(dim=-1)

        samplePoints = randomDepth*coords_inCamera_Flat_Homo # [N,3]

        validIdx = samplePoints[:,2]>0
        samplePoints_World = (self.R.T@(samplePoints-self.t).T).T

        return samplePoints_World, validIdx

    def sampleRayFixDepth(self, dFix:float):
        coords_inImage = torch.stack(torch.meshgrid([torch.arange(0,self.width),torch.arange(0,self.height)],indexing='ij'),dim=-1)
        coords_inImage_Flat = coords_inImage.reshape([-1,2])
        coords_inImage_Flat_Homo = torch.hstack([coords_inImage_Flat,torch.ones(coords_inImage_Flat.shape[0]).unsqueeze(dim=-1)]).to(dtype=torch.float32,device=self.device)
        coords_inCamera_Flat_Homo = (torch.linalg.inv(self.intrinsic)@coords_inImage_Flat_Homo.T).T

        randomDepth = dFix

        samplePoints = randomDepth*coords_inCamera_Flat_Homo

        validIdx = samplePoints[:, 2] > 0
        samplePoints_World = (self.R.T @ (samplePoints - self.t).T).T

        return samplePoints_World, validIdx

def readColmapPoints(path):
    try:
        pointsFile = os.path.join(path, "sparse/0", "points3D.bin")
        points = read_points3D_binary(pointsFile)
    except:
        pointsFile = os.path.join(path, "sparse/0", "points3D.txt")
        points = read_points3D_text(pointsFile)

    xyzs = torch.tensor(points[0])
    rgbs = torch.tensor(points[1])
    errors = torch.tensor(points[2])
    points3d = Point3D(xyzs, rgbs, errors)

    return points3d

class Point3D:
    def __init__(self, xyzs, rgbs, errors):
        self.device = "cuda"
        self.xyzs = xyzs.to(device=self.device)
        self.rgbs = rgbs.to(device=self.device)
        self.errors = errors.to(device=self.device)
        pass

def getDOIfromCams(cameras: List[Camera]):
    R_wc = []
    t_wc = []

    for cam in cameras:
        R_cw = cam.R
        t_cw = cam.t

        R_wci = R_cw.T
        C_i = -R_cw.T @ t_cw

        R_wc.append(R_wci)
        t_wc.append(C_i)

    R_wc = torch.stack(R_wc)
    t_wc = torch.stack(t_wc)

    R3 = R_wc[:, :, 2]

    # Means
    c_mean = R3.mean(dim=0)
    t_mean = t_wc.mean(dim=0)

    # a_i and b_i
    a = R3 - c_mean
    b = t_wc - t_mean

    # Quadratic coefficients
    A = (a * a).sum()
    B = (a * b).sum()

    eps = 1e-12
    if A.abs() < eps:
        Z = torch.tensor(0.0, dtype=R3.dtype)
    else:
        Z = -B / A

    return Z


def getPOIfromCamsZ(cameras: List[Camera], Z):
    R_wc = []
    t_wc = []

    for cam in cameras:
        R_cw = cam.R
        t_cw = cam.t
        R_wci = R_cw.T             # camera→world
        C_i = -R_cw.T @ t_cw       # camera center

        R_wc.append(R_wci)
        t_wc.append(C_i)

    R_wc = torch.stack(R_wc)
    t_wc = torch.stack(t_wc)

    R3 = R_wc[:, :, 2]             # (N,3)

    p = R3 * Z + t_wc              # (N,3)
    p_mean = p.mean(dim=0)
    p_var = ((p - p_mean) ** 2).sum() / len(cameras)
    p_std = p_var ** 0.5

    return p_mean, p_std

# def triangulate_unconstrained(cameras: List, eps: float = 1e-12) -> Tuple[torch.Tensor, dict]:
#     device = cameras[0].R.device
#     dtype = cameras[0].R.dtype
#
#     Cs = []
#     ds = []
#
#     for cam in cameras:
#         R_cw = cam.R
#         t_cw = cam.t
#         R_wc = R_cw.T
#         C = -R_cw.T @ t_cw
#
#         d = R_wc[:, 2]
#         d = d / (d.norm() + 1e-20)
#
#         Cs.append(C)
#         ds.append(d)
#
#     Cs = torch.stack(Cs, dim=0)   # (N,3)
#     ds = torch.stack(ds, dim=0)   # (N,3)
#     N = Cs.shape[0]
#
#     I = torch.eye(3, device=device, dtype=dtype)
#
#     A = torch.zeros((3,3), device=device, dtype=dtype)
#     b = torch.zeros((3,), device=device, dtype=dtype)
#     for i in range(N):
#         d = ds[i].unsqueeze(1)
#         P = I - (d @ d.T)
#         A += P
#         b += P @ Cs[i]
#
#     try:
#         X = torch.linalg.solve(A, b)
#     except RuntimeError:
#         A_reg = A + eps * torch.eye(3, device=device, dtype=dtype)
#         X = torch.linalg.lstsq(A_reg, b.unsqueeze(1)).solution.squeeze()
#
#     if not torch.isfinite(X).all():
#         X = torch.linalg.pinv(A + eps * torch.eye(3, device=device, dtype=dtype)) @ b
#
#     diffs = X.unsqueeze(0) - Cs            # (N,3)
#     s_star = (diffs * ds).sum(dim=1)       # (N,)
#     per_cam_closest = Cs + s_star.unsqueeze(1) * ds
#
#     residuals = (X.unsqueeze(0) - per_cam_closest).norm(dim=1)
#
#     info = {
#         'A': A,
#         'b': b,
#         'per_cam_closest': per_cam_closest,
#         'residuals': residuals
#     }
#
#     return X, info