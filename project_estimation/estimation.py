import cv2
import numpy as np

width = 0.07
height = 0.07

object_points = np.array([
    [0, 0, 0],  # 左下 - 原点
    [width, 0, 0],  # 右下
    [width, height, 0],  # 右上
    [0, height, 0]  # 左上
], dtype=np.float32)

camera_matrix = np.array([
    [8.855441144881335e+02, 0, 6.594432857279190e+02],  # fx, 0, cx
    [0, 8.803756986421753e+02, 3.793912846109970e+02],  # 0, fy, cy
    [0, 0, 1]  # 0, 0, 1
], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

cap = cv2.VideoCapture(0)

def sort_points(points):
    points = points[points[:, 1].argsort()]

    bottom_points = points[2:]
    top_points = points[:2]

    bottom_points = bottom_points[bottom_points[:, 0].argsort()]
    top_points = top_points[top_points[:, 0].argsort()]

    return np.array([
        bottom_points[0],  # 左下
        bottom_points[1],  # 右下
        top_points[1],  # 右上
        top_points[0]  # 左上
    ])


while True:
    ret, frame = cap.read()
    if not ret:
        print("无法获取视频帧")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    image_points = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            image_points = approx.reshape(4, 2).astype(np.float32)
            image_points = sort_points(image_points)
            break

    if image_points is not None:
        for i, p in enumerate(image_points):
            cv2.circle(frame, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
            cv2.putText(frame, str(i), (int(p[0]) + 10, int(p[1]) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


        success, rvec, tvec = cv2.solvePnP(object_points, image_points,
                                           camera_matrix, dist_coeffs,
                                           flags=cv2.SOLVEPNP_ITERATIVE)
        if success:
            axis_length_x = width * 1.5
            axis_length_y = height * 1.5
            axis_length_z = width * 1.2

            axis_points = np.array([
                [0, 0, 0],  # 坐标系原点 (左下角)
                [axis_length_x, 0, 0],  # X轴末端
                [0, axis_length_y, 0],  # Y轴末端
                [0, 0, -axis_length_z]  # Z轴末端
            ], dtype=np.float32)

            projected_points, _ = cv2.projectPoints(axis_points, rvec, tvec,
                                                    camera_matrix, dist_coeffs)
            projected_points = projected_points.reshape(-1, 2).astype(int)

            origin = tuple(projected_points[0])
            x_end = tuple(projected_points[1])
            y_end = tuple(projected_points[2])
            z_end = tuple(projected_points[3])


            cv2.line(frame, origin, x_end, (0, 0, 255), 4)
            cv2.line(frame, origin, y_end, (0, 255, 0), 4)
            cv2.line(frame, origin, z_end, (255, 0, 0), 4)

            arrow_size = 10
            cv2.arrowedLine(frame, origin, x_end, (0, 0, 255), 4, tipLength=0.1)
            cv2.arrowedLine(frame, origin, y_end, (0, 255, 0), 4, tipLength=0.1)
            cv2.arrowedLine(frame, origin, z_end, (255, 0, 0), 4, tipLength=0.1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2

            x_text_pos = (x_end[0] + 10, x_end[1] + 5)
            cv2.putText(frame, "X", x_text_pos, font, font_scale, (0, 0, 255), thickness)

            y_text_pos = (y_end[0] + 5, y_end[1] - 10)
            cv2.putText(frame, "Y", y_text_pos, font, font_scale, (0, 255, 0), thickness)

            z_text_pos = (z_end[0] + 5, z_end[1] + 10)
            cv2.putText(frame, "Z", z_text_pos, font, font_scale, (255, 0, 0), thickness)

            cv2.putText(frame, "Origin (0,0,0)",
                        (origin[0] - 50, origin[1] - 15),
                        font, 0.5, (255, 255, 255), 1)
    cv2.imshow('Pose Estimation', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()