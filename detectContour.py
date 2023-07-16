import cv2
import numpy as np
import matplotlib.pyplot as plt


def build_sample_image():
    """绘制一些基本形状"""
    img = np.ones((500, 500, 3), dtype="uint8") * 70
    cv2.rectangle(img, (50, 50), (250, 250), (255, 0, 255), -1)
    cv2.rectangle(img, (100, 100), (200, 200), (70, 70, 70), -1)
    cv2.circle(img, (350, 350), 100, (255, 255, 0), -1)
    cv2.circle(img, (350, 350), 50, (70, 70, 70), -1)
    return img

def draw_contour_outline(img, cnts, color, thickness=1):
    for cnt in cnts:
        cv2.drawContours(img, [cnt], 0, color, thickness)


# 加载图像并转换为灰度图像
image = build_sample_image()
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用 cv2.threshold() 函数获取二值图像
ret, thresh = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)


# 轮廓检测
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours2, hierarchy2 = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours3, hierarchy3 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# 打印使用不同 mode 参数获得的轮廓数
print("detected contours (RETR_EXTERNAL): '{}' ".format(len(contours)))
print("detected contours (RETR_LIST): '{}' ".format(len(contours2)))
print("detected contours (RETR_TREE): '{}' ".format(len(contours3)))

image_contours = image.copy()
image_contours_2 = image.copy()

# 绘制检测到的轮廓
draw_contour_outline(image_contours, contours, (0, 0, 255), 5)
draw_contour_outline(image_contours_2, contours2, (255, 0, 0), 5)

# 可视化函数
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(1, 4, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

# 可视化
show_img_with_matplotlib(image, "image", 1)
show_img_with_matplotlib(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), "threshold = 100", 2)
show_img_with_matplotlib(image_contours, "contours (RETR EXTERNAL)", 3)
show_img_with_matplotlib(image_contours_2, "contours (RETR LIST)", 4)

plt.show()