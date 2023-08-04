import cv2
import numpy as np
import dlib


def shape_to_np(dlib_shape, dtype="int"):
    # 初始化 (x, y) 坐标列表
    coordinates = np.zeros((dlib_shape.num_parts, 2), dtype=dtype)
    # 循环所有面部特征点，并将其转换为 (x, y) 坐标的元组
    for i in range(0, dlib_shape.num_parts):
        coordinates[i] = (dlib_shape.part(i).x, dlib_shape.part(i).y)
    # 返回 (x,y) 坐标的列表
    return coordinates


# 加载图像并转换为灰度图像
test_face = cv2.imread("img/renwu1.jpg")
gray = cv2.cvtColor(test_face, cv2.COLOR_BGR2GRAY)
# 人脸检测
detector = dlib.get_frontal_face_detector()
rects = detector(gray, 0)

# 第二种面部特征点检测方法，第一行代码
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
# 第二种面部特征点检测方法，第二行代码
shape = predictor(gray, rects[0])
shape = shape_to_np(shape)


# 定义不同特征点取值切片
JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_BRIDGE_POINTS = list(range(27, 31))
LOWER_NOSE_POINTS = list(range(31, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))
ALL_POINTS = list(range(0, 68))
# 使用线条绘制面部特征点
def draw_shape_lines_all(np_shape, image):
    draw_shape_lines_range(np_shape, image, JAWLINE_POINTS)
    draw_shape_lines_range(np_shape, image, RIGHT_EYEBROW_POINTS)
    draw_shape_lines_range(np_shape, image, LEFT_EYEBROW_POINTS)
    draw_shape_lines_range(np_shape, image, NOSE_BRIDGE_POINTS)
    draw_shape_lines_range(np_shape, image, LOWER_NOSE_POINTS)
    draw_shape_lines_range(np_shape, image, RIGHT_EYE_POINTS, True)
    draw_shape_lines_range(np_shape, image, LEFT_EYE_POINTS, True)
    draw_shape_lines_range(np_shape, image, MOUTH_OUTLINE_POINTS, True)
    draw_shape_lines_range(np_shape, image, MOUTH_INNER_POINTS, True)
# 连接不同的点来绘制曲线形状
def draw_shape_lines_range(np_shape, image, range_points, is_closed=False):
    np_shape_display = np_shape[range_points]
    points = np.array(np_shape_display, dtype=np.int32)
    cv2.polylines(image, [points], is_closed, (255, 255, 0), thickness=2, lineType=cv2.LINE_8)
# 函数调用
draw_shape_lines_all(shape, test_face)
cv2.imshow("Landmarks detection using dlib", test_face)
cv2.waitKey(0)
