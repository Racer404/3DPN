import open3d.visualization.gui as gui
import utils
import torch
import cv2
from learnablePerlin3D import PerlinNoise3D
from matplotlib import pyplot as plt


def testCorner(canvas, corners, cam:utils.Camera):
    coords_inCamera = (cam.R @ corners.T).T + cam.t  # (N, 3)
    coords_inImage = cam.intrinsic @ coords_inCamera.T  # (3, N)
    z = coords_inImage[-1]  # (N,)
    mark = (coords_inImage / z)[:2].round().long()  # (2, N)

    x, y = mark[0], mark[1]

    # filter out invalid coordinates
    mask = (x >= 0) & (x < cam.width) & (y >= 0) & (y < cam.height)
    x, y = x[mask], y[mask]

    for idx in enumerate(x):
        cv2.circle(canvas, (x.cpu().numpy()[idx[0]], y.cpu().numpy()[idx[0]]), 3, (0, 0, 255), -1)







if __name__ == "__main__":
    dataset = "kitchen"
    points = utils.readColmapPoints(dataset)
    cams = utils.readColmapSceneInfo(dataset)

    perlin = PerlinNoise3D(scale=1, res=3, device="cuda")
    # for cam in cams:
    #     # dFix = 10
    #     dClose, dFar = cam.getDepthRange()
    #     samplePoints, validPoints = cam.sampleVolumeRandDepth(dClose, dFar)
    #     print(cam.t)  # DEBUG
    #     renderedPoints = perlin.getValue(samplePoints, validPoints)
    #     output = renderedPoints.reshape(389, 260, 1).squeeze()
    #     image = output.T.cpu().detach().numpy()
    #
    #     plt.imshow(image)
    #     plt.show()

    cornerCoords = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.float64, device="cuda")
    toPerlinCenter = torch.tensor([0.5, 0.5, 0.5]).to(dtype=torch.float64, device="cuda")
    testCenter = torch.tensor([-0.461083, 1.5, 1.5],dtype=torch.float64, device="cuda")

    drawCorner = (cornerCoords - toPerlinCenter) + testCenter



    for cam in cams:
        # make sure height, width order is correct
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

        out = image.cpu().detach().numpy()
        testCorner(out,drawCorner,cam)

        plt.imshow(out)
        plt.show()