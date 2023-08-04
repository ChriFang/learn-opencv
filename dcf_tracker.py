import cv2
import dlib

def draw_text_info():
    # 绘制文本的位置
    menu_pos_1 = (10, 20)
    menu_pos_2 = (10, 40)
    # 绘制菜单信息
    cv2.putText(frame, "Use '1' to re-initialize tracking", menu_pos_1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    if tracking_face:
        cv2.putText(frame, "tracking the face", menu_pos_2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    else:
        cv2.putText(frame, "detecting a face to initialize tracking...", menu_pos_2, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255))

# 创建视频捕获对象
capture = cv2.VideoCapture(0)
# 加载人脸检测器
detector = dlib.get_frontal_face_detector()
# 初始化追踪器
tracker = dlib.correlation_tracker()
# 当前是否在追踪人脸
tracking_face = False

while True:
    # 捕获视频帧
    ret, frame = capture.read()
    # 绘制基本信息
    draw_text_info()
    
    if tracking_face is False:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 尝试检测人脸以初始化跟踪器
        rects = detector(gray, 0)
        # 通过判断是否检测到人脸来决定是否启动追踪
        if len(rects) > 0:
            # Start tracking:
            tracker.start_track(frame, rects[0])
            tracking_face = True

    if tracking_face is True:
        # 更新跟踪器并打印测量跟踪器的置信度
        print(tracker.update(frame))
        # 获取被跟踪对象的位置
        pos = tracker.get_position()
        # 绘制被跟踪对象的位置
        cv2.rectangle(frame, (int(pos.left()), int(pos.top())), (int(pos.right()), int(pos.bottom())), (0, 255, 0), 3)
    # 捕获键盘事件
    key = 0xFF & cv2.waitKey(1)

    # 按 1 初始化追踪器
    if key == ord("1"):
        tracking_face = False
    # 按 q 退出
    if key == ord('q'):
        break
    # 显示结果
    cv2.imshow("Face tracking using dlib frontal face detector and correlation filters for tracking", frame)
# 释放所有资源
capture.release()
cv2.destroyAllWindows()
