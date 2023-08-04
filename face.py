import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_detection(image, faces):
    """在每个检测到的人脸上绘制一个矩形进行标示"""
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5)
    return image


# 可视化函数
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(2, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')
    
# 加载图像
img = cv2.imread("img/renwu2.jpg")
# 将 BGR 图像转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 加载分类器
cas_alt2 = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt2.xml")
cas_default = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
# 检测人脸
faces_alt2 = cas_alt2.detectMultiScale(gray)
faces_default = cas_default.detectMultiScale(gray)
retval, faces_haar_alt2 = cv2.face.getFacesHAAR(img, "haarcascades/haarcascade_frontalface_alt2.xml")
faces_haar_alt2 = np.squeeze(faces_haar_alt2)
if faces_haar_alt2.ndim == 1:
    faces_haar_alt2.resize(1,4)
retval, faces_haar_default = cv2.face.getFacesHAAR(img, "haarcascades/haarcascade_frontalface_default.xml")
faces_haar_default = np.squeeze(faces_haar_default)
if faces_haar_default.ndim == 1:
    faces_haar_default.resize(1,4)

# 绘制人脸检测框
img_faces_alt2 = show_detection(img.copy(), faces_alt2)
img_faces_default = show_detection(img.copy(), faces_default)
img_faces_haar_alt2 = show_detection(img.copy(), faces_haar_alt2)
img_faces_haar_default = show_detection(img.copy(), faces_haar_default)

# 可视化
show_img_with_matplotlib(img_faces_alt2, "detectMultiScale(frontalface_alt2): " + str(len(faces_alt2)), 1)
show_img_with_matplotlib(img_faces_default, "detectMultiScale(frontalface_default): " + str(len(faces_default)), 2)
show_img_with_matplotlib(img_faces_haar_alt2, "getFacesHAAR(frontalface_alt2): " + str(len(faces_haar_alt2)), 3)
show_img_with_matplotlib(img_faces_haar_default, "getFacesHAAR(frontalface_default): " + str(len(faces_haar_default)), 4)
plt.show()
