import cv2
import numpy as np
import matplotlib.pyplot as plt


labels = np.array([1, 1, -1, -1, -1])
data = np.matrix([[800, 40], [850, 400], [500, 10], [550, 300], [450, 600]], dtype=np.float32)


def svm_init(C=12.5, gamma=0.50625):
    """ 创建 SVM 模型并为其分配主要参数，返回模型 """
    model = cv2.ml.SVM_create()
    model.setGamma(gamma)
    model.setC(C)
    model.setKernel(cv2.ml.SVM_LINEAR)
    model.setType(cv2.ml.SVM_C_SVC)
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    return model
# 初始化 SVM 模型
svm_model = svm_init(C=12.5, gamma=0.50625)


def svm_train(model, samples, responses):
    # 使用 samples 和 responses 训练模型
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    return model
# 训练 SVM
svm_train(svm_model, data, labels)

def svm_predict(model, samples):
    """根据训练好的模型预测响应"""

    return model.predict(samples)[1].ravel()

def show_svm_response(model, image):

    colors = {1: (255, 255, 0), -1: (0, 255, 255)}

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            sample = np.matrix([[j, i]], dtype=np.float32)
            response = svm_predict(model, sample)

            image[i, j] = colors[response.item(0)]
    
    cv2.circle(image, (800, 40), 10, (255, 0, 0), -1)
    cv2.circle(image, (850, 400), 10, (255, 0, 0), -1)

    cv2.circle(image, (500, 10), 10, (0, 255, 0), -1)
    cv2.circle(image, (550, 300), 10, (0, 255, 0), -1)
    cv2.circle(image, (450, 600), 10, (0, 255, 0), -1)

    support_vectors = model.getUncompressedSupportVectors()
    for i in range(support_vectors.shape[0]):
        cv2.circle(image, (int(support_vectors[i, 0]), int(support_vectors[i, 1])), 15, (0, 0, 255), 6)

# 创建图像
img_output = np.zeros((640, 1200, 3), dtype="uint8")
# 显示 SVM 响应
show_svm_response(svm_model, img_output)

cv2.imshow("svm predict", img_output)
cv2.waitKey(0)
cv2.destroyAllWindows()
