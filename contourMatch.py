import cv2
import numpy as np
import matplotlib.pyplot as plt

def build_circle_image():
    """绘制参考圆"""
    img = np.zeros((500, 500, 3), dtype="uint8")
    cv2.circle(img, (250, 250), 200, (255, 255, 255), 1)
    return img

def get_position_to_draw(text, point, font_face, font_scale, thickness):
    """获取图形坐标中心点"""
    text_size = cv2.getTextSize(text, font_face, font_scale, thickness)[0]
    text_x = point[0] - text_size[0] / 2
    text_y = point[1] + text_size[1] / 2
    return round(text_x), round(text_y)

def show_img_with_matplotlib(color_img, title, pos):
    """图像可视化"""
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

# 加载图像
image = cv2.imread("polygon.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_circle = build_circle_image()
gray_image_circle = cv2.cvtColor(image_circle, cv2.COLOR_BGR2GRAY)

# 二值化图像
ret, thresh = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY_INV)
ret, thresh_circle = cv2.threshold(gray_image_circle, 70, 255, cv2.THRESH_BINARY)

# 检测轮廓
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours_circle, hierarchy_2 = cv2.findContours(thresh_circle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

result_1 = image.copy()
result_2 = image.copy()
result_3 = image.copy()

for contour in contours:
    # 计算轮廓的矩
    M = cv2.moments(contour)

    # 计算矩的质心
    cX = 0
    cY = 0
    if M['m00'] != 0:
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
   # else:
   #     print("m00 is zero")

    # 使用三种匹配模式将每个轮廓与圆形轮廓进行匹配
    ret_1 = cv2.matchShapes(contours_circle[0], contour, cv2.CONTOURS_MATCH_I1, 0.0)
    ret_2 = cv2.matchShapes(contours_circle[0], contour, cv2.CONTOURS_MATCH_I2, 0.0)
    ret_3 = cv2.matchShapes(contours_circle[0], contour, cv2.CONTOURS_MATCH_I3, 0.0)

    # 将获得的分数写在结果图像中
    (x_1, y_1) = get_position_to_draw(str(round(ret_1, 3)), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    (x_2, y_2) = get_position_to_draw(str(round(ret_2, 3)), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    (x_3, y_3) = get_position_to_draw(str(round(ret_3, 3)), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)

    cv2.putText(result_1, str(round(ret_1, 3)), (x_1+10, y_1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(result_2, str(round(ret_2, 3)), (x_2+10, y_2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(result_3, str(round(ret_3, 3)), (x_3+10, y_3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

show_img_with_matplotlib(result_1, "result 1", 1)
show_img_with_matplotlib(result_2, "result 2", 2)
show_img_with_matplotlib(result_3, "result 3", 3)

plt.show()