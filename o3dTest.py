import threading

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import torch
from matplotlib import pyplot as plt

import learnablePerlin3D
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


def my_render_function(cam, extrinsic, perlinNoise):
    cam.R = torch.tensor(extrinsic[:3,:3], device="cuda", dtype=torch.float64)
    cam.t = torch.tensor(extrinsic[:3,3], device="cuda", dtype=torch.float64)

    dClose, dFar = cam.getDepthRange(perlinNoise)
    samplePoints_Volume, validPoints = cam.sampleVolumeBySteps(dClose, dFar, dSteps)
    renderedPoints_Volume, output_mask_Volume = perlin.getValue(samplePoints_Volume, validPoints)

    renderedPoints_Flat = renderedPoints_Volume.reshape(cam.width * cam.height, dSteps)
    renderedPoints = renderedPoints_Flat @ dAlpha

    output = renderedPoints.reshape(cam.width, cam.height)
    image = output.T.cpu().detach().numpy()

    colormap = plt.get_cmap('viridis')
    colored_image = colormap(image)
    return colored_image

def render_pointCloud(cam, extrinsic, points):
    cam.R = torch.tensor(extrinsic[:3, :3], device="cuda", dtype=torch.float64)
    cam.t = torch.tensor(extrinsic[:3, 3], device="cuda", dtype=torch.float64)

    image = torch.zeros([cam.height, cam.width, 3], dtype=torch.uint8, device="cuda")

    coords_inCamera = (cam.R @ points.xyzs.T).T + cam.t  # (N, 3)
    coords_inImage = cam.intrinsic @ coords_inCamera.T  # (3, N)
    z = coords_inImage[-1]  # (N,)
    mark = (coords_inImage / z)[:2].round().long()  # (2, N)

    x, y = mark[0], mark[1]

    # filter out invalid coordinates
    mask = (x >= 0) & (x < cam.width) & (y >= 0) & (y < cam.height)
    x, y = x[mask], y[mask]

    colors = points.rgbs[mask].to(torch.uint8).to(image.device)
    # paint
    image[y, x] = colors

    return image.cpu().detach().numpy()

class CustomCameraViewer:
    def __init__(self, cam:utils.Camera, perlin:learnablePerlin3D.PerlinNoise3D, points:utils.Point3D):
        self.cam = cam
        self.fov = 50.0
        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window("Custom Camera Renderer", cam.width * 3, cam.height)

        # Set up scene widget (left side)
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene_widget)

        # Add simple geometry so the camera can look at something
        box = o3d.geometry.TriangleMesh.create_box(width=perlin.scale, height=perlin.scale, depth=perlin.scale)
        box.translate(-box.get_center())
        box.translate(perlin.center.cpu())

        box.compute_vertex_normals()
        self.scene_widget.scene.add_geometry("box", box, rendering.MaterialRecord())

        # Set up camera based on geometry
        bbox = box.get_axis_aligned_bounding_box()
        self.scene_widget.setup_camera(self.fov, bbox, bbox.get_center())

        # Set up image view widget (middle side)
        self.image_panel = gui.ImageWidget()
        self.window.add_child(self.image_panel)

        # Set up image view widget (right side)
        self.image_panel_B = gui.ImageWidget()
        self.window.add_child(self.image_panel_B)

        # Layout: side-by-side
        def on_layout(ctx):
            r = self.window.content_rect
            half = r.width // 3
            self.scene_widget.frame = gui.Rect(r.x, r.y, half, r.height)
            self.image_panel.frame = gui.Rect(r.x + half, r.y, half, r.height)
            self.image_panel_B.frame = gui.Rect(r.x + 2 * half, r.y, half, r.height)
        self.window.set_on_layout(on_layout)

        # Start render loop
        gui.Application.instance.post_to_main_thread(self.window, self.update_loop)

    def update_loop(self):
        # Get extrinsic (4x4 world-to-camera)
        extrinsic = np.array(self.scene_widget.scene.camera.get_view_matrix(), dtype=np.float64)
        ToGLCamera = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        extrinsic =  ToGLCamera @ extrinsic

        # Call your custom rendering function
        img = my_render_function(self.cam, extrinsic, perlin)
        # Convert to uint8 for display
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
        o3d_img = o3d.t.geometry.Image(img_uint8)
        self.image_panel.update_image(o3d_img)

        pointCloud_out = render_pointCloud(self.cam, extrinsic, points)
        o3d_pointCloud = o3d.t.geometry.Image(pointCloud_out)
        self.image_panel_B.update_image(o3d_pointCloud)

        # Schedule next update (e.g., every 0.1s)
        threading.Timer(0.1, lambda: gui.Application.instance.post_to_main_thread(self.window, self.update_loop)).start()

    def run(self):
        self.app.run()


if __name__ == "__main__":
    dataset = "kitchen"
    cams = utils.readColmapSceneInfo(dataset)
    points = utils.readColmapPoints(dataset)
    cam = cams[0]
    dSteps = 100  # Ray sampling frequency
    dAlpha = utils.smoothStepsFunc(dSteps).to(device=cam.device)

    perlin = PerlinNoise3D(scale=1, res=3, device="cuda", center=torch.tensor([-0.461083, 1.5, 1.5], dtype=torch.float64))

    viewer = CustomCameraViewer(cam, perlin, points)
    viewer.run()
