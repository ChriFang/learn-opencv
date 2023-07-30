import cv2
import numpy as np

# 加载图像
image_query = cv2.imread('img/sunflower-1.png')
image_scene = cv2.imread('img/sunflower.png')
# ORB 检测器初始化
orb = cv2.ORB_create()
# 使用 ORB 检测关键点并计算描述符
keypoints_1, descriptors_1 = orb.detectAndCompute(image_query, None)
keypoints_2, descriptors_2 = orb.detectAndCompute(image_scene, None)

bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

bf_matches = bf_matcher.match(descriptors_1, descriptors_2)

bf_matches = sorted(bf_matches, key=lambda x: x.distance)


# result = cv2.drawMatches(image_1, keypoints_1, image_2, keypoints_2, bf_matches[:20], None, matchColor=(255, 255, 0), singlePointColor=(255, 0, 255), flags=0)

# 提取匹配的关键点
pts_src = np.float32([keypoints_1[m.queryIdx].pt for m in bf_matches]).reshape(-1, 1, 2)
pts_dst = np.float32([keypoints_2[m.trainIdx].pt for m in bf_matches]).reshape(-1, 1, 2)
# 计算单应性矩阵
M, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)

# 获取“查询”图像的角坐标
h, w = image_query.shape[:2]
pts_corners_src = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
# 使用矩阵 M 和“查询”图像的角点执行透视变换，以获取“场景”图像中“检测”对象的角点：
pts_corners_dst = cv2.perspectiveTransform(pts_corners_src, M)

img_obj = cv2.polylines(image_scene, [np.int32(pts_corners_dst)], True, (0, 255, 255), 10)

img_matching = cv2.drawMatches(image_query, keypoints_1, img_obj, keypoints_2, bf_matches, None, matchColor=(255, 255, 0), singlePointColor=(255, 0, 255), flags=0)

cv2.imshow("key points", img_matching)
cv2.waitKey(0)
cv2.destroyAllWindows()