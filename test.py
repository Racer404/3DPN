import os
import threading
import time

import cv2
import copy
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import optim

import utils
from learnablePerlin3D import PerlinNoise3D


def compute_intrinsic_matrix(fov_deg, width, height):
    fov_rad = np.deg2rad(fov_deg)
    fx = fy = 0.5 * width / np.tan(fov_rad / 2)
    cx = width / 2
    cy = height / 2
    return np.array([
        [fx,  0, cx],
        [0,  fy, cy],
        [0,   0,  1]
    ], dtype=np.float64)

class PerlinViewer:
    def __init__(self, Cam:utils.Camera, Perlin:PerlinNoise3D, Points:utils.Point3D):
        self.cam = Cam
        self.perlin = Perlin
        self.points = Points
        self.dSteps = 100
        self.dAlpha = utils.smoothStepsFunc(self.dSteps).to(device=self.cam.device)

        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window("Custom Camera Renderer", self.cam.width * 3, self.cam.height)

        self.scene_box = gui.SceneWidget()
        self.scene_perlin = gui.ImageWidget()
        self.scene_pointcloud = gui.ImageWidget()
        self.window.add_child(self.scene_box)
        self.window.add_child(self.scene_perlin)
        self.window.add_child(self.scene_pointcloud)

        self.scene_box.scene = rendering.Open3DScene(self.window.renderer)
        # Set up camera based on geometry
        box = o3d.geometry.TriangleMesh.create_box(width=self.perlin.scale, height=self.perlin.scale, depth=self.perlin.scale)
        box.translate(-box.get_center())
        box.translate(self.perlin.center.cpu())
        box.compute_vertex_normals()
        self.scene_box.scene.add_geometry("box", box, rendering.MaterialRecord())
        bbox = box.get_axis_aligned_bounding_box()
        self.scene_box.setup_camera(50., bbox, bbox.get_center())

        # Layout: side-by-side
        def on_layout(ctx):
            r = self.window.content_rect
            half = r.width // 3
            self.scene_box.frame = gui.Rect(r.x, r.y, half, r.height)
            self.scene_perlin.frame = gui.Rect(r.x + half, r.y, half, r.height)
            self.scene_pointcloud.frame = gui.Rect(r.x + 2 * half, r.y, half, r.height)
        self.window.set_on_layout(on_layout)

    def render_perlin(self):
        dClose, dFar = self.cam.getDepthRange(self.perlin)
        samplePoints_Volume, validPoints = self.cam.sampleVolumeBySteps(dClose, dFar, self.dSteps)
        renderedPoints_Volume, output_mask_Volume = self.perlin.getValue(samplePoints_Volume, validPoints)
        renderedPoints_Volume[~output_mask_Volume] = 0.5

        renderedPoints_Flat = renderedPoints_Volume.reshape(self.cam.width * self.cam.height, self.dSteps)
        renderedPoints = renderedPoints_Flat @ self.dAlpha

        output_mask_Flat = output_mask_Volume.reshape(self.cam.width * self.cam.height, self.dSteps)
        output_mask = torch.any(output_mask_Flat, dim=1)

        # renderedPoints[~output_mask] = 0.5  # Background
        output = renderedPoints.reshape(self.cam.width, self.cam.height)
        image = output.T.cpu().detach().numpy()

        colormap = plt.get_cmap('viridis')
        colored_image = colormap(image)

        return (colored_image * 255).astype(np.uint8)

    def render_pointCloud(self):
        image = torch.zeros([self.cam.height, self.cam.width, 3], dtype=torch.uint8, device="cuda")
        coords_inCamera = (self.cam.R @ self.points.xyzs.T).T + self.cam.t  # (N, 3)
        coords_inImage = self.cam.intrinsic @ coords_inCamera.T  # (3, N)
        z = coords_inImage[-1]  # (N,)
        mark = (coords_inImage / z)[:2].round().long()  # (2, N)
        x, y = mark[0], mark[1]
        # filter out invalid coordinates
        mask = (x >= 0) & (x < self.cam.width) & (y >= 0) & (y < self.cam.height)
        x, y = x[mask], y[mask]
        colors = self.points.rgbs[mask].to(torch.uint8).to(image.device)
        # paint
        image[y, x] = colors

        return image.cpu().detach().numpy()

    def start_loop(self):
        # Kick off the recurring loop once GUI starts
        self.app.post_to_main_thread(self.window, self.on_loop)

    def on_loop(self):
        # Get camera transform
        extrinsic = np.array(self.scene_box.scene.camera.get_view_matrix(), dtype=np.float64)
        ToGLCamera = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        extrinsic = ToGLCamera @ extrinsic
        self.cam.R = torch.tensor(extrinsic[:3, :3], device="cuda", dtype=torch.float64)
        self.cam.t = torch.tensor(extrinsic[:3, 3], device="cuda", dtype=torch.float64)

        # Render both images
        img_perlin = self.render_perlin()
        o3d_img_perlin = o3d.t.geometry.Image(img_perlin)

        img_pointCloud = self.render_pointCloud()
        o3d_img_pointCloud = o3d.t.geometry.Image(img_pointCloud)

        # Safe, direct update (no thread issues)
        self.scene_perlin.update_image(o3d_img_perlin)
        self.scene_pointcloud.update_image(o3d_img_pointCloud)

        # Schedule next frame (~30 FPS)
        time.sleep(1 / 30.0)
        self.app.post_to_main_thread(self.window, self.on_loop)


def train(
        perlin: PerlinNoise3D = None,
        cameras: [utils.Camera] = None,
        iterations: int = None,
        lr: float = None,
        ifVisualize: bool = False,
        ifSaveResult: bool = False) -> torch.Tensor:
    optimizer = optim.Adam(
        [perlin.cornerVecs], lr
    )
    mse_loss = torch.nn.MSELoss()
    mae_loss = torch.nn.L1Loss()

    output_img = None
    frames = []
    perlin.loss = []
    dSteps = 100
    dAlpha = utils.smoothStepsFunc(dSteps).to(device=cams[0].device)

    for iter in range(iterations):
        for cam in cameras:
            dClose, dFar = cam.getDepthRange(perlin)
            samplePoints_Volume, validPoints = cam.sampleVolumeBySteps(dClose, dFar, dSteps)
            renderedPoints_Volume, output_mask_Volume = perlin.getValue(samplePoints_Volume, validPoints)
            renderedPoints_Volume[~output_mask_Volume] = 0.5

            renderedPoints_Flat = renderedPoints_Volume.reshape(cam.width * cam.height, dSteps)
            renderedPoints = renderedPoints_Flat @ dAlpha

            output_mask_Flat = output_mask_Volume.reshape(cam.width * cam.height, dSteps)
            output_mask = torch.any(output_mask_Flat, dim=1)

            gtImage = (torch.tensor(cv2.imread(cam.image,cv2.IMREAD_GRAYSCALE), dtype=torch.float64, device="cuda")/255.).T
            gtImage_Flat = gtImage.flatten()
            loss = mse_loss(renderedPoints[output_mask], gtImage_Flat[output_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")
            perlin.loss.append(loss.item())

            if ifVisualize:
                # renderedPoints[~output_mask] = 0.5 #Background
                output_img = renderedPoints.reshape(cam.width, cam.height)
                torch.cuda.synchronize()
                showImg = output_img.T.cpu().detach().numpy()
                showGt = gtImage.T.cpu().detach().numpy()
                cv2.imshow("Training", showImg)
                cv2.imshow("GT", showGt)
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

    return output_img

if __name__ == "__main__":
    dataset = "kitchen"
    cams = utils.readColmapSceneInfo(dataset)
    points = utils.readColmapPoints(dataset)
    testCenter = torch.tensor([-0.461083, 1.5, 1.5], dtype=torch.float64, device="cuda")
    perlin = PerlinNoise3D(scale=2, res=20, center=testCenter, device="cuda")

    train(perlin, cams, 100, 0.01, False, False)

    viewerCam = cams[0]
    viewer = PerlinViewer(viewerCam,perlin,points)
    viewer.start_loop()
    viewer.app.run()