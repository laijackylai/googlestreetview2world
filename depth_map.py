import numpy as np
import cv2 as cv
from matplotlib import cm, pyplot as plt
import sys

np.set_printoptions(suppress=True, threshold=sys.maxsize)

# imgL = cv2.imread('./img/gsv_0.jpg',0)
# imgR = cv2.imread('./img/gsv_1.jpg',0)

# stereo = cv2.StereoBM_create(numDisparities=20, blockSize=16)
# disparity = stereo.compute(imgL,imgR)
# cv2.imwrite('./depth_maps/depth_map.png', disparity)

width = 0
height = 0
data = []

with open('depth.txt') as data:
    for index, line in enumerate(data):
        if index == 0:
            width = int(line.strip())
        if index == 1:
            height = int(line.strip())
        if index == 2:
            data = line.strip()

data = np.reshape(np.fromstring(data, dtype=np.float32, sep=','), (height, width))
data = 1/(1 + np.exp(-data))

plt.imshow(data, cmap='plasma')
plt.colorbar()
plt.savefig('./depth.png')
