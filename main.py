import cv2
import numpy as np
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't retrive frame")
        break
    cv2.imshow("Feed", frame)
    cv2.waitKey(1)
