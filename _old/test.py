import torch
import cv2
from matplotlib import pyplot as plt

mountain_file = cv2.imread("../mountain.png", cv2.IMREAD_GRAYSCALE)
mountain = torch.tensor(mountain_file,device="cuda",dtype=torch.float64)/255.
lenna_file = cv2.imread("../Lenna.jpg", cv2.IMREAD_GRAYSCALE)
lenna = torch.tensor(lenna_file,device="cuda",dtype=torch.float64)/255.
blocks_file = cv2.imread("../blocks.jpg", cv2.IMREAD_GRAYSCALE)
blocks = torch.tensor(blocks_file,device="cuda",dtype=torch.float64)/255.

randImg = torch.rand([640,640])

testImg = mountain.cpu()
plt.figure("input")
plt.imshow(testImg)

fft_x = torch.fft.rfft(testImg, dim = 1)
magnitude_spectrum_x = torch.log(torch.abs(fft_x) + 1e-9)
plt.figure("magnitude_spectrum_x")
plt.imshow(magnitude_spectrum_x)
N_x = testImg.shape[1]

fft_x_mean = magnitude_spectrum_x.mean(dim = 0)
plt.figure("avg_fft_x")
plt.plot(fft_x_mean[1:])
sorted_indices_x = torch.argsort(fft_x_mean, descending=True)
print("TOP 10 x freq:")
for i in range(10):
    idx = sorted_indices_x[i].item()
    if (idx == 0):
        print("average intensity: " + str(fft_x_mean[idx].item()))
    else:
        print(N_x/idx)

fft_y = torch.fft.rfft(testImg.cpu(), dim = 0)
magnitude_spectrum_y = torch.log(torch.abs(fft_y) + 1e-9)
plt.figure("magnitude_spectrum_y")
plt.imshow(magnitude_spectrum_y)
N_y = testImg.shape[0]

fft_y_mean = magnitude_spectrum_y.mean(dim = 1)
plt.figure("avg_fft_y")
plt.plot(fft_y_mean[1:])
sorted_indices_y = torch.argsort(fft_y_mean, descending=True)
print("TOP 10 y freq:")
for i in range(10):
    idx = sorted_indices_y[i].item()
    if(idx == 0):
        print("average intensity: "+str(fft_y_mean[idx].item()))
    else:
        print(N_y/idx)


#
# fft_xy = torch.fft.fft(fft_x, dim = 0)
# plt.figure("magnitude_spectrum_xy")
# magnitude_spectrum_xy = torch.log(torch.abs(fft_xy) + 1e-9)
# plt.imshow(magnitude_spectrum_xy)

plt.show()
breakpoint()
