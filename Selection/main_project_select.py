import cv2
import numpy as np

def detect_L_shape_comprehensive(image_path):
    image = cv2.imread(image_path)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_light_blue = np.array([95, 50, 200])
    upper_light_blue = np.array([120, 100, 255])
    color_mask = cv2.inRange(hsv, lower_light_blue, upper_light_blue)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, brightness_mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)

    combined_mask = cv2.bitwise_or(color_mask, brightness_mask)

    kernel_L = np.ones((3, 3), np.uint8)
    processed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_L, iterations=3)

    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 150 < area < 2200:
            epsilon = 0.03 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            vertice = len(approx)

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            rect = cv2.minAreaRect(contour)
            width, height = rect[1]

            is_L_shape = (0.4 < solidity < 0.8) and (min(width, height) > 5) and (4 <=vertice <= 6)

            if is_L_shape:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return image, processed_mask


result, mask = detect_L_shape_comprehensive('image.png')
cv2.imshow('Detection', result)
cv2.imshow('Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()