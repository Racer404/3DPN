import torch
import cv2

imgFile = cv2.imread("../mountain.png", cv2.IMREAD_GRAYSCALE)
groundTruth = (torch.tensor(imgFile,device="cuda",dtype=torch.float32)-127.5)/127.5

breakpoint()
