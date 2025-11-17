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
    downsample = 8.
    images_path = [path + "/images_8/" + cam_extrinsics[i].name for i in range(n)]
    raw = cv2.imread(images_path[0])
    h,w = raw.shape[0:2]
    intrinsic = torch.tensor(
        [[cam_intrinsics[1].params[0]/downsample,0,cam_intrinsics[1].params[2]/downsample],
         [0,cam_intrinsics[1].params[1]/downsample,cam_intrinsics[1].params[3]/downsample],
         [0,0,1]]).to(dtype=torch.float64)
    poses_R = torch.tensor(numpy.array([cam_extrinsics[i].qvec2rotmat() for i in range(n)]))
    poses_t = torch.tensor(numpy.array([cam_extrinsics[i].tvec for i in range(n)]))

    cameras = [Camera(width=w,height=h,intrinsic=intrinsic,R=poses_R[i],t=poses_t[i],image=images_path[i]) for i in range(n)]
    return cameras


def smoothStepsFunc(steps):
    steps_n = torch.arange(steps, dtype=torch.float64) / steps
    steps_n = torch.cat([steps_n[1:], torch.tensor([1])])
    func = -(steps_n ** 2) + 2 * steps_n  # Looking for a function from [0,1] -> [0,1]
    alpha = func - torch.cat([torch.tensor([0]), func[:-1]])
    return alpha

def sigmoid_ix(x, i):
    return 1 / (1 + torch.exp(-i * x))

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
        lastRow = torch.tensor([0,0,0,1], dtype=torch.float64, device=self.device)
        upperPart = torch.concatenate([self.R, self.t.unsqueeze(1)], dim=1)
        mat4_4_Rt = torch.concatenate([upperPart, lastRow.unsqueeze(0)])
        return mat4_4_Rt


    def getDepthRange(self, center, scale):
        cornerCoords = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.float64, device=self.device)
        toPerlinCenter = torch.tensor([0.5, 0.5, 0.5]).to(dtype=torch.float64, device=self.device)

        cornerCoords = scale * (cornerCoords - toPerlinCenter) + center
        cornerCoords_Camera = (self.R @ cornerCoords.T).T + self.t
        cornerDepth_Camera =  cornerCoords_Camera[:,2]

        return cornerDepth_Camera.min().item(), cornerDepth_Camera.max().item()

    def sampleVolumeBySteps(self, dClose:float, dFar:float, steps: int):
        coords_inImage = torch.stack(torch.meshgrid([torch.arange(0,self.width),torch.arange(0,self.height)],indexing='ij'),dim=-1)
        coords_inImage_Flat = coords_inImage.reshape([-1,2])
        coords_inImage_Flat_Homo = torch.hstack([coords_inImage_Flat,torch.ones(coords_inImage_Flat.shape[0]).unsqueeze(dim=-1)]).to(dtype=torch.float64,device=self.device)
        coords_inCamera_Flat_Homo = (torch.linalg.inv(self.intrinsic)@coords_inImage_Flat_Homo.T).T

        steps_norm = torch.arange(0, 1, 1 / steps, dtype=torch.float64)
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
        coords_inImage_Flat_Homo = torch.hstack([coords_inImage_Flat,torch.ones(coords_inImage_Flat.shape[0]).unsqueeze(dim=-1)]).to(dtype=torch.float64,device=self.device)
        coords_inCamera_Flat_Homo = (torch.linalg.inv(self.intrinsic)@coords_inImage_Flat_Homo.T).T

        randomDepth = (torch.rand(coords_inCamera_Flat_Homo.shape[0],device=self.device)*(dFar-dClose)+dClose).unsqueeze(dim=-1)

        samplePoints = randomDepth*coords_inCamera_Flat_Homo # [N,3]

        validIdx = samplePoints[:,2]>0
        samplePoints_World = (self.R.T@(samplePoints-self.t).T).T

        return samplePoints_World, validIdx

    def sampleRayFixDepth(self, dFix:float):
        coords_inImage = torch.stack(torch.meshgrid([torch.arange(0,self.width),torch.arange(0,self.height)],indexing='ij'),dim=-1)
        coords_inImage_Flat = coords_inImage.reshape([-1,2])
        coords_inImage_Flat_Homo = torch.hstack([coords_inImage_Flat,torch.ones(coords_inImage_Flat.shape[0]).unsqueeze(dim=-1)]).to(dtype=torch.float64,device=self.device)
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

def getPOI(cameras: List[Camera]): #Need further improvement
    Rs = torch.stack([cam.R for cam in cameras])
    R_3 = Rs[:,:,2]
    t = torch.stack([cam.t for cam in cameras])

    u_R = torch.mean(R_3, dim = 0)
    u_t = torch.mean(t, dim = 0)

    optimalZ = torch.sum((R_3-u_R)*(t-u_t),dim=0) / torch.sum((R_3-u_R)**2,dim = 0)
    var_cPtoMean = torch.sum((torch.mean((optimalZ*R_3 + t), dim = 0) - (optimalZ*R_3 + t))**2, dim = 0) / len(cameras)

    breakpoint()
    return optimalZ
