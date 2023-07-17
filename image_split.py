import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_with_matplotlib(color_img, title, pos):
    # Convert BGR image to RGB
    img_RGB = color_img[:,:,::-1]

    ax = plt.subplot(3, 6, pos)
    plt.imshow(img_RGB)
    plt.title(title,fontsize=8)
    plt.axis('off')

image = cv2.imread('xiaogou.png')

image_without_blue = image.copy()
image_without_blue[:, :, 0] = 0

image_without_green = image.copy()
image_without_green[:, :, 1] = 0

image_without_red = image.copy()
image_without_red[:, :, 2] = 0


# 使用缩放因子
resized_image = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_AREA)

# 平移，x方向平移200像素，y方向平移30像素
height, width = image.shape[:2]
M = np.float32([[1, 0, 200], [0, 1, 30]])
dst_image_move = cv2.warpAffine(image, M, (width, height))


# 旋转30度
M = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 30, 1)
dst_rot = cv2.warpAffine(image, M, (width, height))


plt.figure(figsize=(13,5))
plt.suptitle('splitting and merging channels in OpenCV', fontsize=12, fontweight='bold')

show_with_matplotlib(image, "BGR - image", 1)
show_with_matplotlib(image_without_blue, "BGR without B", 2)
show_with_matplotlib(image_without_green, "BGR without G", 3)
show_with_matplotlib(image_without_red, "BGR without R", 4)
show_with_matplotlib(resized_image, "image scale 2x", 5)
show_with_matplotlib(dst_image_move, "image move", 6)
show_with_matplotlib(dst_rot, "image rotation", 8)

plt.show()
