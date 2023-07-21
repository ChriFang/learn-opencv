import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_contour_outline(img, cnts, color, thickness=1):
    for cnt in cnts:
        cv2.drawContours(img, [cnt], 0, color, thickness)

# 加载图像并将其转化为灰度图像
image = cv2.imread("cat.jpeg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 获取二值图像
ret, thresh = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)
thresh = cv2.bitwise_not(thresh)
# 计算图像矩，传递参数为图像
M = cv2.moments(thresh, True)
print("moments: '{}'".format(M))
def centroid(moments):
    """根据图像矩计算质心"""
    x_centroid = 0
    y_centroid = 0
    if moments['m00'] != 0:
        x_centroid = round(moments['m10'] / moments['m00'])
        y_centroid = round(moments['m01'] / moments['m00'])
    return x_centroid, y_centroid
# 计算质心
x, y = centroid(M)
# 计算 Hu 矩并打印
HuM = cv2.HuMoments(M)
print("Hu moments: '{}'".format(HuM))

# 计算图像矩时传递轮廓，重复以上计算过程
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
M2 = cv2.moments(contours[0])
# print(contours)
print("moments: '{}'".format(M2))
x2, y2 = centroid(M2)
# 绘制轮廓
draw_contour_outline(image, contours, (255, 0, 0), 10)
# 绘制质心
cv2.circle(image, (x, y), 10, (255, 255, 0), -1)
# cv2.circle(image, (x2, y2), 10, (0, 0, 255), -1)
# 打印质心
print("('x','y'): ('{}','{}')".format(x, y))
print("('x2','y2'): ('{}','{}')".format(x2, y2))
# 可视化，show_img_with_matplotlib()函数与前述示例类似，不再赘述
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(1, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')
def show_thresh_with_matplotlib(thresh, title, pos):
    ax = plt.subplot(1, 2, pos)
    plt.imshow(thresh, cmap='gray')
    plt.title(title, fontsize=8)
    plt.axis('off')
show_img_with_matplotlib(image, "detected contour and centroid", 1)
show_thresh_with_matplotlib(thresh, 'thresh', 2)
plt.show()
