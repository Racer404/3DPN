import os
import numpy
import utils
import torch
import cv2
import learnablePerlin3D
from matplotlib import  pyplot as plt

dataset = "kitchen"
cams = utils.readColmapSceneInfo(dataset)
resultRefCam = cams[10]

mainFolder = "LNPL Data analysis/"

targetSetups = ["directScale_shuffle_bceLogit/", "directScale_shuffle_mse/", "sig10^2.2fx_shuffle_mse/",
                "sige^1autoScale(x)_shuttle_mse/", "sige^1minmax(x)_shuffle_mse/", "sige^1x_shuffle_mse/", "sige^10x_shuffle_mse/"]

# targetSetups = ["test/"]

for targetSetup in targetSetups:
    p30 = learnablePerlin3D.readTensor(mainFolder+targetSetup+"0.pth")

    dClose, dFar = resultRefCam.getDepthRange(p30.center, p30.scale)
    dAlpha = utils.smoothStepsFunc(100).to(device=resultRefCam.device)

    samplePoints_Volume, validPoints = resultRefCam.sampleVolumeBySteps(dClose, dFar, 100)

    renderedPoints_Volume, output_mask_Volume = p30.getValue(samplePoints_Volume, validPoints)

    renderedPoints_Flat = renderedPoints_Volume.reshape(resultRefCam.width * resultRefCam.height, 100)
    renderedPoints = renderedPoints_Flat @ dAlpha

    gtImage = (torch.tensor(cv2.imread(resultRefCam.image,cv2.IMREAD_GRAYSCALE), dtype=torch.float64, device="cuda")/255.).T
    gtImage_Flat = gtImage.flatten()

    output_mask_Flat = output_mask_Volume.reshape(resultRefCam.width * resultRefCam.height, 100)
    output_mask = torch.any(output_mask_Flat, dim=1)

    output_img = renderedPoints.reshape(resultRefCam.width, resultRefCam.height)
    torch.cuda.synchronize()
    showImg = (output_img.T.cpu().detach().numpy() * 255).astype(numpy.uint8)
    showGt = (gtImage.T.cpu().detach().numpy() * 255).astype(numpy.uint8)

    save_dir = mainFolder+targetSetup

    cv2.imwrite(os.path.join(save_dir, "trained.png"), showImg)
    cv2.imwrite(os.path.join(save_dir, "gt.png"), showGt)

    # === Save histograms and plots ===
    def save_plot(fig_name):
        """Helper function to save current matplotlib figure."""
        plt.savefig(os.path.join(save_dir, fig_name))
        plt.close()

    # volumeValid
    volumeValid = renderedPoints_Volume[output_mask_Volume].cpu()

    plt.figure()
    plt.hist(volumeValid, bins=50)
    save_plot("volumeValid_hist.png")

    plt.figure()
    plt.plot(volumeValid)
    save_plot("volumeValid_raw.png")

    # renderedValid & gtValid
    renderedValid = renderedPoints[output_mask].cpu()
    gtValid = gtImage_Flat[output_mask].cpu()

    plt.figure()
    plt.hist(renderedValid, bins=50)
    save_plot("renderedValid_hist.png")

    plt.figure()
    plt.hist(gtValid, bins=50)
    save_plot("gtValid_hist.png")

    plt.figure()
    plt.plot(renderedValid)
    save_plot("renderedValid_raw.png")

    plt.figure()
    plt.plot(gtValid)
    save_plot("gtValid_raw.png")

    # plotCornerMean
    plotCornerMean = p30.cornerVecs.mean(dim=1).cpu()

    plt.figure()
    plt.hist(plotCornerMean, bins=50)
    save_plot("trainedCornerMean_hist.png")

    plt.figure()
    plt.plot(plotCornerMean)
    save_plot("trainedCornerMean_raw.png")