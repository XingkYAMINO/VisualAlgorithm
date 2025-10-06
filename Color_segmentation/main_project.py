import cv2
import numpy as np

img_bgr = cv2.imread("bed_pic.png")
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
img_h, img_s, img_v = cv2.split(img_hsv)
mask_pink_h1 = cv2.inRange(img_h, 150, 180)
mask_pink_s = cv2.inRange(img_s, 50, 200)
mask_pink = cv2.bitwise_and(mask_pink_h1  , mask_pink_s)
img_out = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_pink)

cv2.imshow("Result", img_out)
cv2.imwrite("result.png", img_out)
cv2.waitKey(0)