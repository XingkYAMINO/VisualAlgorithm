import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

width = 0.07
height = 0.07

object_points = np.array([
    [0, 0, 0],  # 左下 - 原点
    [width, 0, 0],  # 右下 (X正方向)
    [width, height, 0],  # 右上
    [0, height, 0]  # 左上 (Y正方向)
], dtype=np.float32)

camera_matrix = np.array([
    [8.855441144881335e+02, 0, 6.594432857279190e+02],
    [0, 8.803756986421753e+02, 3.793912846109970e+02],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.array([
    -0.1336,  # k1
    0.4863,  # k2
    0.0,  # p1
    0.0,  # p2
    0.0  # k3
], dtype=np.float32).reshape(5, 1)


def auto_canny(image, sigma=0.33):
    md = np.median(image)
    lower_value = int(max(0, (1.0 - sigma) * md))
    upper_value = int(min(255, (1.0 + sigma) * md))
    return cv.Canny(image, lower_value, upper_value)


def order_points_correctly(points):

    points = points.reshape(4, 2)

    y_sorted = points[np.argsort(points[:, 1])]  # 按y坐标升序排列

    bottom_points = y_sorted[2:]
    # 在下方点中，x较小的为左下，x较大的为右下
    bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]
    left_bottom = bottom_points[0]  # 左下
    right_bottom = bottom_points[1]  # 右下
    top_points = y_sorted[:2]
    # 在上方点中，x较大的为右上，x较小的为左上
    top_points = top_points[np.argsort(top_points[:, 0])]
    left_top = top_points[0]  # 左上
    right_top = top_points[1]  # 右上

    ordered = np.array([left_bottom, right_bottom, right_top, left_top], dtype=np.float32)
    return ordered

def improve_corner_detection(image):

    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    img_blur = cv.GaussianBlur(img_gray, (5, 5), 0)

    thresh_adaptive = cv.adaptiveThreshold(img_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv.THRESH_BINARY, 11, 2)

    edges = auto_canny(thresh_adaptive)

    kernel = np.ones((3, 3), np.uint8)
    edges_enhanced = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

    return edges_enhanced

#角点检测
def find_corner_improved(image):

    edges_enhanced = improve_corner_detection(image)

    contours, _ = cv.findContours(edges_enhanced, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:5]

    for contour in contours:
        # 过滤小面积轮廓
        if cv.contourArea(contour) < 1000:
            continue

        perimeter = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:
            points = approx.reshape(4, 2)

            # 计算四边形质量
            area = cv.contourArea(approx)
            if area < 1000:  # 面积太小可能不可靠
                continue

            # 凸性检查
            if not cv.isContourConvex(approx):
                continue

            return points.astype(np.float32)

    return None


def precise_corner_refinement(image, rough_corners):

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined_corners = cv.cornerSubPix(gray, rough_corners, (5, 5), (-1, -1), criteria)

    return refined_corners


while True:
    ret, frame = cap.read()
    if not ret:
        print("无法获取帧")
        break

    display_frame = frame.copy()
    img_points = find_corner_improved(frame)

    if img_points is not None:
        img_points_refined = precise_corner_refinement(frame, img_points)

        img_points_ordered = order_points_correctly(img_points_refined)

        for i, (x, y) in enumerate(img_points_ordered):
            cv.circle(display_frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            labels = ["LB", "RB", "RT", "LT"]
            cv.putText(display_frame, labels[i], (int(x) + 10, int(y) + 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv.polylines(display_frame, [img_points_ordered.astype(np.int32)], True, (0, 255, 255), 2)

        success, rvec, tvec = cv.solvePnP(object_points, img_points_ordered, camera_matrix, dist_coeffs)

        if success:

            cv.drawFrameAxes(display_frame, camera_matrix, dist_coeffs,rvec, tvec, width * 0.5)

    cv.imshow('Improved Pose Estimation', display_frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()