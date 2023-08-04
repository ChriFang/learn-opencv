import cv2
import numpy as np
import matplotlib.pyplot as plt


# 可视化函数
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')


# 加载图像
image = cv2.imread("img/renwu2.jpg")
# 加载预训练的模型， Caffe 实现的版本
net = cv2.dnn.readNetFromCaffe("dnn_model/deploy.prototxt", "dnn_model/res10_300x300_ssd_iter_140000_fp16.caffemodel")
# 加载预训练的模型， Tensorflow 实现的版本
# net = cv2.dnn.readNetFromTensorflow("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")

blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104., 117., 123.], False, False)

# 将 blob 设置为输入并获取检测结果
net.setInput(blob)
detections = net.forward()


detected_faces = 0
w, h = image.shape[1], image.shape[0]
# 迭代所有检测结果
for i in range(0, detections.shape[2]):
    # 获取当前检测结果的置信度
    confidence = detections[0, 0, i, 2]
    # 如果置信大于最小置信度，则将其可视化
    if confidence > 0.7:
        detected_faces += 1
        # 获取当前检测结果的坐标
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')
        # 绘制检测结果和置信度
        text = "{:.3f}%".format(confidence * 100)
        y = startY -10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 3)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

# 可视化
show_img_with_matplotlib(image, "DNN face detector: " + str(detected_faces), 1)
plt.show()

