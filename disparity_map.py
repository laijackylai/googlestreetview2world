import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('./raw_img/gsv_0.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./raw_img/gsv_2.jpg', cv2.IMREAD_GRAYSCALE)

# * plot original images alongside each other
# fig, axes = plt.subplots(1, 2, figsize=(15, 10))
# axes[0].imshow(img1, cmap="gray")
# axes[1].imshow(img2, cmap="gray")
# axes[0].axhline(250)
# axes[1].axhline(250)
# axes[0].axhline(450)
# axes[1].axhline(450)
# plt.suptitle("Original images")
# plt.savefig('original_images.png', dpi=500)

# * detecting keypoints
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# * plot key points on original images
# imgSift1 = cv2.drawKeypoints(
#     img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imwrite("./key_points/keypoints_0.png", imgSift1)

# imgSift2 = cv2.drawKeypoints(
#     img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imwrite("./key_points/keypoints_2.png", imgSift2)

# * match key points in both images
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

matchesMask = [[0, 0] for i in range(len(matches))]
good = []
pts1 = []
pts2 = []

for i, (m, n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        # Keep this keypoint pair
        matchesMask[i] = [1, 0]
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

# * visualize good key points
# draw_params = dict(matchColor=(0, 255, 0),
#                    singlePointColor=(255, 0, 0),
#                    matchesMask=matchesMask,
#                    flags=cv2.DrawMatchesFlags_DEFAULT)

# keypoint_matches = cv2.drawMatchesKnn(
#     img1, kp1, img2, kp2, matches, None, **draw_params)
# cv2.imwrite("Keypoint_matches.png", keypoint_matches)

# * stereo rectification
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
fundamental_matrix, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

pts1 = pts1[inliers.ravel() == 1]
pts2 = pts2[inliers.ravel() == 1]

# * visualize epilines
def drawlines(img1src, img2src, lines, pts1src, pts2src):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1src.shape
    img1color = cv2.cvtColor(img1src, cv2.COLOR_GRAY2BGR)
    img2color = cv2.cvtColor(img2src, cv2.COLOR_GRAY2BGR)
    # Edit: use the same random seed so that two images are comparable!
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1color = cv2.line(img1color, (x0, y0), (x1, y1), color, 1)
        img1color = cv2.circle(img1color, tuple(pt1), 5, color, -1)
        img2color = cv2.circle(img2color, tuple(pt2), 5, color, -1)
    return img1color, img2color

lines1 = cv2.computeCorrespondEpilines(
    pts2.reshape(-1, 1, 2), 2, fundamental_matrix)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

lines2 = cv2.computeCorrespondEpilines(
    pts1.reshape(-1, 1, 2), 1, fundamental_matrix)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

plt.subplot(121), plt.imshow(img5)
plt.subplot(122), plt.imshow(img3)
plt.suptitle("Epilines in both images")
plt.savefig('epilines.png', dpi=500)

# * stereo rectification (uncalibrated)
h1, w1 = img1.shape
h2, w2 = img2.shape
_, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1))
img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1))
img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2))
img1_rectified = cv2.rotate(img1_rectified, cv2.ROTATE_90_CLOCKWISE)
img2_rectified = cv2.rotate(img2_rectified, cv2.ROTATE_90_CLOCKWISE)
# cv2.imwrite("./rectified/rectified_1.png", img1_rectified)
# cv2.imwrite("./rectified/rectified_2.png", img2_rectified)

rows, cols = img2_rectified.shape
M = np.float32([[1, 0, 20], [0, 1, 0]])
left_img = img1_rectified
right_img = cv2.warpAffine(img2_rectified, M, (cols, rows))

# * draw the rectified images
# fig, axes = plt.subplots(1, 2, figsize=(15, 10))
# axes[0].imshow(img1_rectified, cmap="gray")
# axes[1].imshow(img2_rectified, cmap="gray")
# axes[0].axhline(200)
# axes[1].axhline(200)
# axes[0].axhline(400)
# axes[1].axhline(400)
# axes[0].axvline(200)
# axes[1].axvline(200)
# axes[0].axvline(400)
# axes[1].axvline(400)
# plt.savefig("rectified_images.png", dpi=500)

# * calculate disparity
block_size = 5
min_disp = -16
max_disp = 16
num_disp = max_disp - min_disp
uniquenessRatio = 5
speckleWindowSize = 200
speckleRange = 2
disp12MaxDiff = 0

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    uniquenessRatio=uniquenessRatio,
    speckleWindowSize=speckleWindowSize,
    speckleRange=speckleRange,
    disp12MaxDiff=disp12MaxDiff,
    P1=8 * 1 * block_size * block_size,
    P2=32 * 1 * block_size * block_size,
)
disparity_SGBM = stereo.compute(left_img, right_img)
disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                              beta=0, norm_type=cv2.NORM_MINMAX)
disparity_SGBM = np.uint8(disparity_SGBM)
cv2.imwrite("./depth_map/disparity_SGBM_norm.png", disparity_SGBM)

# * previous code
# block_size = 9
# min_disp = -128
# max_disp = 128
# num_disp = max_disp - min_disp
# uniquenessRatio = 5
# speckleWindowSize = 200
# speckleRange = 2
# disp12MaxDiff = 0

# stereo = cv2.StereoSGBM_create(
#     minDisparity=min_disp,
#     numDisparities=num_disp,
#     blockSize=block_size,
#     uniquenessRatio=uniquenessRatio,
#     speckleWindowSize=speckleWindowSize,
#     speckleRange=speckleRange,
#     disp12MaxDiff=disp12MaxDiff,
#     P1=8 * 1 * block_size * block_size,
#     P2=32 * 1 * block_size * block_size,
# )
# disparity_SGBM = stereo.compute(img1, img2)

# # Normalize the values to a range from 0..255 for a grayscale image
# disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
#                               beta=0, norm_type=cv2.NORM_MINMAX)
# disparity_SGBM = np.uint8(disparity_SGBM)

# cv2.imwrite("./depth_map/disparity_SGBM_norm.png", disparity_SGBM)
