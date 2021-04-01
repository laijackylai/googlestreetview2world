import numpy as np
import cv2

img = cv2.imread('./raw_img/gsv_2.jpg')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gaussian = cv2.GaussianBlur(imgray, (3, 3), 0)
bilateral = cv2.bilateralFilter(imgray, 9, 75, 75)

cv2.imwrite('./bilateral/bilateral_2.png', bilateral)
cv2.imwrite('./gaussian/gaussian_2.png', gaussian)

# * best so far: bilateral filtering -> gaussian mean thresholding
ret1, thresh1 = cv2.threshold(bilateral, 70, 255, 0)
thresh2 = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
thresh3 = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
ret4, thresh4 = cv2.threshold(bilateral, 0 ,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(thresh3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contour_img = cv2.drawContours(np.zeros((np.shape(imgray)[0],np.shape(imgray)[1]), float), contours, -1, (255, 255, 255), 1)
cv2.imwrite('./contour/contour_2.png', contour_img)
