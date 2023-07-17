import cv2
import argparse

capture = cv2.VideoCapture(0)
if capture.isOpened() is False:
    print("Error opening the camera")
frame_index = 0
while capture.isOpened():
    ret, frame = capture.read()

    if ret is True:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 保存相机画面
        if cv2.waitKey(20) & 0xFF == ord('c'):
            frame_name = "/Users/fangruibin/Documents/code/test/opencv/camera_frame_{}.png".format(frame_index)
            gray_frame_name = "/Users/fangruibin/Documents/code/test/opencv/grayscale_camera_frame_{}.png".format(frame_index)
            # 将当前帧保存到磁盘(同时保存 BGR 和灰度帧)
            cv2.imwrite(frame_name, frame)
            cv2.imwrite(gray_frame_name, gray_frame)
            frame_index += 1
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        cv2.imshow('Input frame from the camera', frame)
    else:
        break

capture.release()
cv2.destroyAllWindows()
