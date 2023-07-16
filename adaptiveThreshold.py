import cv2
import numpy as np
import matplotlib.pyplot as plt


# 可视化函数
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

# 加载图像
image = cv2.imread('girl.jpeg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 绘制灰度图像
show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "img", 1)

# 双边滤波
gray_image = cv2.bilateralFilter(gray_image, 15, 25, 25)
# 自适应阈值处理
thresh1 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

show_img_with_matplotlib(cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR), "method=THRESH_MEAN_C, blockSize=11, C=2", 2)

plt.show()