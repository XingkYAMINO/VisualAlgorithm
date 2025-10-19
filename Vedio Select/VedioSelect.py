import cv2
import numpy as np

def detect_Vedio(video_path, output_path=None):
    """
    检测视频中的花露水瓶

    参数:
    video_path: 输入视频路径
    output_path: 输出视频路径(可选)
    """

    # 打开视频
    cap = cv2.VideoCapture("Bottle.mp4")

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 定义绿色范围 (HSV颜色空间)
    # 根据花露水的实际颜色调整这些值
    lower_green = np.array([35, 50, 50])    # 较低的HSV绿色阈值
    upper_green = np.array([85, 255, 255])  # 较高的HSV绿色阈值

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"处理第 {frame_count} 帧")

        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 创建绿色掩码
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # 形态学操作去除噪声
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 处理每个轮廓
        for contour in contours:
            # 过滤小面积噪声
            area = cv2.contourArea(contour)
            if area < 500:  # 根据实际情况调整这个阈值
                continue

            # 方法1: 最小外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # 方法2: 多边形拟合
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # 方法3: 凸包
            hull = cv2.convexHull(contour)

            # 在原图上绘制检测结果

            # 绘制最小外接矩形 (绿色)
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

            # 绘制多边形拟合 (蓝色)
            cv2.drawContours(frame, [approx], 0, (255, 0, 0), 2)

            # 绘制凸包 (红色)
            cv2.drawContours(frame, [hull], 0, (0, 0, 255), 2)

            # 添加标签
            cv2.putText(frame, "Min Area Rect", tuple(box[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "Polygon Fit", tuple(approx[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, "Convex Hull", tuple(hull[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 显示结果
        cv2.imshow('Hualu Shui Detection', frame)

        # 保存输出视频
        if output_path:
            out.write(frame)

        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
# 使用方法
if __name__ == "__main__":
    # 方法1: 处理视频
    video_path = "Bottle.mp4"  # 替换为您的视频路径
    output_path = "output_video.mp4"

    detect_Vedio(video_path, output_path)

