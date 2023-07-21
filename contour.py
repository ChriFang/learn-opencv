import cv2
import numpy as np
import matplotlib.pyplot as plt

def extreme_points(contour):
    """检测轮廓的极值点"""

    extreme_left = tuple(contour[contour[:, :, 0].argmin()][0])
    extreme_right = tuple(contour[contour[:, :, 0].argmax()][0])
    extreme_top = tuple(contour[contour[:, :, 1].argmin()][0])
    extreme_bottom = tuple(contour[contour[:, :, 1].argmax()][0])

    return extreme_left, extreme_right, extreme_top, extreme_bottom

def array_to_tuple(arr):
    """将列表转换为元组"""
    return tuple(arr.reshape(1, -1)[0])

def draw_contour_points(img, cnts, color):
    """绘制所有检测到的轮廓点"""
    for cnt in cnts:
        squeeze = np.squeeze(cnt)
        for p in squeeze:
            pp = array_to_tuple(p)
            cv2.circle(img, pp, 10, color, -1)
    return img

def draw_contour_outline(img, cnts, color, thickness=1):
    """绘制所有轮廓"""
    for cnt in cnts:
        cv2.drawContours(img, [cnt], 0, color, thickness)

def show_img_with_matplotlib(color_img, title, pos):
    """图像可视化"""
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

# 加载图像并转换为灰度图像
image = cv2.imread("cat.jpeg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 阈值处理转化为二值图像
ret, thresh = cv2.threshold(gray_image, 60, 255, cv2.THRESH_BINARY)
# thresh = cv2.bitwise_not(thresh)

# 利用二值图像检测图像中的轮廓
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 显示检测到的轮廓数
print("detected contours: {} ".format(len(contours)))
# 创建原始图像的副本以执行可视化
boundingRect_image = image.copy()
minAreaRect_image = image.copy()
fitEllipse_image = image.copy()
minEnclosingCircle_image = image.copy()
approxPolyDP_image = image.copy()

# 1. cv2.boundingRect()
x, y, w, h = cv2.boundingRect(contours[0])
print("rslt: x {} y {} w {} h {} ".format(x, y, w, h))
cv2.rectangle(boundingRect_image, (x, y), (x + w, y + h), (0, 255, 0), 5)
#draw_contour_outline(boundingRect_image, contours, (255, 0, 0), 10)

# 2. cv2.minAreaRect()
rotated_rect = cv2.minAreaRect(contours[0])
box = cv2.boxPoints(rotated_rect)
box = np.intp(box)
cv2.polylines(minAreaRect_image, [box], True, (0, 0, 255), 5)

# 3. cv2.minEnclosingCircle()
(x, y), radius = cv2.minEnclosingCircle(contours[0])
center = (int(x), int(y))
radius = int(radius)
cv2.circle(minEnclosingCircle_image, center, radius, (255, 0, 0), 5)

# 4. cv2.fitEllipse()
# ellipse = cv2.fitEllipse(contours[0])
# cv2.ellipse(fitEllipse_image, ellipse, (0, 255, 255), 5)

# 5. cv2.approxPolyDP()
#epsilon = 0.01 * cv2.arcLength(contours[0], True)
#approx = cv2.approxPolyDP(contours[0], epsilon, True)
#draw_contour_outline(approxPolyDP_image, [approx], (255, 255, 0), 5)
#draw_contour_points(approxPolyDP_image, [approx], (255, 0, 255))

# 检测轮廓的极值点
#left, right, top, bottom = extreme_points(contours[0])
#cv2.circle(image, left, 20, (255, 0, 0), -1)
#cv2.circle(image, right, 20, (0, 255, 0), -1)
#cv2.circle(image, top, 20, (0, 255, 255), -1)
#cv2.circle(image, bottom, 20, (0, 0, 255), -1)

# 可视化
show_img_with_matplotlib(image, "image and extreme points", 1)
show_img_with_matplotlib(boundingRect_image, "cv2.boundingRect()", 2)
show_img_with_matplotlib(minAreaRect_image, "cv2.minAreaRect()", 3)
show_img_with_matplotlib(minEnclosingCircle_image, "cv2.minEnclosingCircle()", 4)
show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray image", 5)
#show_img_with_matplotlib(fitEllipse_image, "cv2.ellipse()", 5)
#show_img_with_matplotlib(approxPolyDP_image, "cv2.approxPolyDP()", 6)
plt.show()
