import numpy
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import torch

import utils
from learnablePerlin3D import PerlinNoise3D


class PerlinViewer:
    def __init__(self, Cam:utils.Camera, Perlins:[PerlinNoise3D], Points:utils.Point3D):
        self.cam = Cam
        self.perlins = Perlins
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
        box = o3d.geometry.TriangleMesh.create_box(width=self.perlins[0].scale, height=self.perlins[0].scale, depth=self.perlins[0].scale)
        box.translate(-box.get_center())
        box.translate(self.perlins[0].center.cpu())
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
        p_close, p_far = self.cam.getDepthRange(self.perlins[0].center, self.perlins[0].scale)
        d_start = p_close if p_close > 0. else 0.00001
        d_end = d_start + self.perlins[0].scale * 1.73205  # 1.73205 ~ sqrt(3)

        requestPoints_Volume = self.cam.sampleVolumeBySteps(d_start, d_end, self.dSteps)[0]
        mask_Volume = utils.maskValidPoints(requestPoints_Volume, self.perlins[0].center, self.perlins[0].scale)

        rendered_perPerlin = torch.stack([p.getValue(requestPoints_Volume[mask_Volume]) for p in self.perlins])
        renderedPoints_Valid = rendered_perPerlin.mean(dim=0)

        renderedPoints_Volume = torch.zeros([requestPoints_Volume.shape[0], self.perlins[0].channelNum], dtype=torch.float64,
                                            device="cuda")
        renderedPoints_Volume[mask_Volume] = renderedPoints_Valid / 2. + 0.5
        renderedPoints_Volume[~mask_Volume] = 0.5

        renderedPoints_Flat = renderedPoints_Volume.reshape(self.cam.width * self.cam.height, self.dSteps, self.perlins[0].channelNum)
        renderedPoints = torch.matmul(renderedPoints_Flat.transpose(1, 2), self.dAlpha)
        pred_img = renderedPoints.reshape(self.cam.width, self.cam.height, self.perlins[0].channelNum)

        showImg = pred_img.transpose(0, 1).contiguous().cpu().detach().numpy()
        showImg = numpy.clip(showImg, 0., 1.)

        return (showImg * 255).astype(np.uint8)

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

        # Render both images　        # Safe, direct update (no thread issues)
        img_perlin = self.render_perlin()
        o3d_img_perlin = o3d.t.geometry.Image(img_perlin)
        self.scene_perlin.update_image(o3d_img_perlin)

        # img_pointCloud = self.render_pointCloud()
        # o3d_img_pointCloud = o3d.t.geometry.Image(img_pointCloud)
        # self.scene_pointcloud.update_image(o3d_img_pointCloud)

        # time.sleep(1 / 30.0) # Schedule next frame (~30 FPS)
        self.app.post_to_main_thread(self.window, self.on_loop)