import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib


# 可视化函数
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')


# 加载图像
image = cv2.imread("img/renwu1.jpg")
cnn_face_detector = dlib.cnn_face_detection_model_v1("model/mmod_human_face_detector.dat")
rects = cnn_face_detector(image, 0)
def show_detection(image, faces):
    """使用矩形检测框显式标示每个检测到的人脸"""
    for face in faces:
        cv2.rectangle(image, (face.rect.left(), face.rect.top()), (face.rect.right(), face.rect.bottom()), (255, 255, 0), 5)
    return image
# 绘制检测框
img_faces = show_detection(image.copy(), rects)
# 可视化
show_img_with_matplotlib(img_faces, "cnn_face_detector(img, 0): " + str(len(rects)), 1)
plt.show()
