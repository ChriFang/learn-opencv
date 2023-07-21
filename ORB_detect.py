import cv2

orb = cv2.ORB_create()
# 加载图像
image = cv2.imread('polygon.jpg')
# 检测图像中的关键点
keypoints = orb.detect(image, None)
keypoints, descriptors = orb.compute(image, keypoints)
image_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(255, 0, 255), flags=0)

cv2.imshow("key points", image_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
