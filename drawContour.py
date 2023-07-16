import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_test_contour():
    cnts = [np.array(
        [[[600, 320]], [[460, 562]], [[180, 563]], [[40, 320]],
         [[179, 78]], [[459, 77]]], dtype=np.int32)]
    return cnts

contours = get_test_contour()
print("contour shape: '{}'".format(contours[0].shape))
print("'detected' contours: '{}' ".format(len(contours)))

def draw_contour_outline(img, cnts, color, thickness=1):
    for cnt in cnts:
        cv2.drawContours(img, [cnt], 0, color, thickness)


def draw_contour_points(img, cnts, color):
    for cnt in cnts:
        # 维度压缩
        squeeze = np.squeeze(cnt)
        # 遍历轮廓阵列的所有点
        for p in squeeze:
            # 为了绘制圆点，需要将列表转换为圆心元组
            p = array_to_tuple(p)
            # 绘制轮廓点
            cv2.circle(img, p, 10, color, -1)

    return img


def array_to_tuple(arr):
    """将列表转换为元组"""
    return tuple(arr.reshape(1, -1)[0])


# 创建画布并复制，用于显示不同检测效果
canvas = np.zeros((640, 640, 3), dtype="uint8")
image_contour_points = canvas.copy()
image_contour_outline = canvas.copy()
image_contour_points_outline = canvas.copy()
# 绘制轮轮廓点
draw_contour_points(image_contour_points, contours, (255, 0, 255))

# 绘制轮廓
draw_contour_outline(image_contour_outline, contours, (0, 255, 255), 3)

# 同时绘制轮廓和轮廓点
draw_contour_outline(image_contour_points_outline, contours, (255, 0, 0), 3)
draw_contour_points(image_contour_points_outline, contours, (0, 0, 255))


# 可视化函数
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(1, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')


# 绘制图像
show_img_with_matplotlib(image_contour_points, "contour points", 1)
show_img_with_matplotlib(image_contour_outline, "contour outline", 2)
show_img_with_matplotlib(image_contour_points_outline, "contour outline and points", 3)

# 可视化
plt.show()
