import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('./img/gsv_0.jpg',0)
imgR = cv2.imread('./img/gsv_1.jpg',0)

stereo = cv2.StereoBM_create(numDisparities=20, blockSize=16)
disparity = stereo.compute(imgL,imgR)
cv2.imwrite('./depth_maps/depth_map.png', disparity)
