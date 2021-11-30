import cv2
import numpy as np
cap = cv2.VideoCapture(0)

#Click to capture the image in the box
def captureImage(event,x,y,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("here")
        capture = params[0][height//3-(int)(height*.10)+2:height//3*2-(int)(height*.10), height//3+3:height//3*2]
        print(capture)
        cv2.imshow("Input Image", capture)
        #cv2.waitKey(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't retrive frame")
        break
    height = frame.shape[1]
    upperLeft = (height//3, height//3-(int)(height*.10))
    bottomRight = (height//3*2, height//3*2-(int)(height*.10))
    cv2.rectangle(frame, upperLeft, bottomRight, (255, 0, 0), (1))
    frame = cv2.flip(frame, 1)
    cv2.namedWindow("Feed")
    cv2.imshow("Feed", frame)
    params = [frame]
    cv2.setMouseCallback("Feed",captureImage, params)
    
    cv2.waitKey(1)
