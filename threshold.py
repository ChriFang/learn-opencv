import cv2
import numpy as np
import matplotlib.pyplot as plt


# 可视化函数
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(3, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

# 加载图像
image = cv2.imread('girl.jpeg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 绘制灰度图像
show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "img", 1)

# 使用不同的阈值调用 cv2.threshold() 并进行可视化
for i in range(8):
    ret, thresh = cv2.threshold(gray_image, 130 + i * 10, 255, cv2.THRESH_BINARY)
    show_img_with_matplotlib(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), "threshold = {}".format(130 + i * 10), i + 2)


plt.show()
