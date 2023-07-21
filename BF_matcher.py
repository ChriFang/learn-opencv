import cv2

# 加载图像
image_1 = cv2.imread('wh300.jpg')
image_2 = cv2.imread('wh300_r.jpg')
# ORB 检测器初始化
orb = cv2.ORB_create()
# 使用 ORB 检测关键点并计算描述符
keypoints_1, descriptors_1 = orb.detectAndCompute(image_1, None)
keypoints_2, descriptors_2 = orb.detectAndCompute(image_2, None)

bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

bf_matches = bf_matcher.match(descriptors_1, descriptors_2)

bf_matches = sorted(bf_matches, key=lambda x: x.distance)

result = cv2.drawMatches(image_1, keypoints_1, image_2, keypoints_2, bf_matches[:20], None, matchColor=(255, 255, 0), singlePointColor=(255, 0, 255), flags=0)

cv2.imshow("key points", result)
cv2.waitKey(0)
cv2.destroyAllWindows()