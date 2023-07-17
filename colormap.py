import cv2
import matplotlib.pyplot as plt


def show_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 7, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

# 加载图像并转化为灰度图像
gray_img = cv2.imread('xiaogou.png', cv2.IMREAD_GRAYSCALE)

# 色彩映射列表
colormaps = ["AUTUMN", "BONE", "JET", "WINTER", "RAINBOW", "OCEAN", "SUMMER", "SPRING", "COOL", "HSV", "HOT", "PINK", "PARULA"]

plt.figure(figsize=(12, 5))
plt.suptitle("Colormaps", fontsize=14, fontweight='bold')

show_with_matplotlib(cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR), "GRAY", 1)

# 应用色彩映射
for idx, val in enumerate(colormaps):
    show_with_matplotlib(cv2.applyColorMap(gray_img, idx), val, idx + 2)

plt.show()
