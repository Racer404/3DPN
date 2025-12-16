import numpy
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import torch
from matplotlib import pyplot as plt
from typing import List

import learnablePerlin3D
import utils
from learnablePerlin3D import PerlinNoise3D


class PerlinViewer:
    def __init__(self, Cam:utils.Camera, Perlin: PerlinNoise3D, optimalZ: float, dSteps: int, Points:utils.Point3D):
        self.cam = Cam
        self.perlin = Perlin
        self.points = Points
        self.dSteps = dSteps
        self.dAlpha = utils.smoothStepsFunc(self.dSteps).to(device=self.cam.device)
        self.optimalZ = optimalZ
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
        box = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
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
        p_close = 0.5
        p_far = self.optimalZ * 2
        requestPoints_Volume = self.cam.sampleVolumeBySteps(p_close, p_far, self.dSteps)[0]
        renderedPoints_Volume = self.perlin.getValue(requestPoints_Volume, None) / 2. + 0.5
        rendered_color = renderedPoints_Volume[:, :-1]
        rendered_alpha = renderedPoints_Volume[:, -1]

        renderedPoints_Flat = utils.renderVolume_stepsRaypass(rendered_color, rendered_alpha, self.dSteps).squeeze()

        pred_img = renderedPoints_Flat.reshape(self.cam.width, self.cam.height, 3)

        pred_numpy = pred_img.transpose(0, 1).contiguous().cpu().detach().numpy()
        pred_numpy = numpy.clip(pred_numpy, 0., 1.)

        return (pred_numpy * 255).astype(np.uint8)

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
        extrinsic = np.array(self.scene_box.scene.camera.get_view_matrix(), dtype=np.float32)
        ToGLCamera = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        extrinsic = ToGLCamera @ extrinsic
        self.cam.R = torch.tensor(extrinsic[:3, :3], device="cuda", dtype=torch.float32)
        self.cam.t = torch.tensor(extrinsic[:3, 3], device="cuda", dtype=torch.float32)

        # Render both images　        # Safe, direct update (no thread issues)
        img_perlin = self.render_perlin()
        o3d_img_perlin = o3d.t.geometry.Image(img_perlin)
        self.scene_perlin.update_image(o3d_img_perlin)

        img_pointCloud = self.render_pointCloud()
        o3d_img_pointCloud = o3d.t.geometry.Image(img_pointCloud)
        self.scene_pointcloud.update_image(o3d_img_pointCloud)

        # time.sleep(1 / 30.0) # Schedule next frame (~30 FPS)
        self.app.post_to_main_thread(self.window, self.on_loop)