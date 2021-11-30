import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D

NNPATH = "./checkpoint/checkpoint.ckpt"#path to NN

cnn_model = None #make the model global
def drawText(text, frame, x, y):
    cv2.putText(frame, text, (x, y), font, scale, color, 2)

def loadNN():#load the model
    layers =  [Input(shape=(28,28,1),name='shape'),#make layers
        Conv2D(16,3,padding="same",activation="relu"),
        MaxPool2D(),
        Conv2D(32,3,padding="same",activation="relu"),
        MaxPool2D(),
        Flatten(),
        Dense(units=26, activation="softmax",name="out")]
    
    tmp = Sequential(layers)
    tmp.load_weights(NNPATH)#load the data
    tmp.summary()#print a summary of the model
    cnn_model = tmp


def predict(img):#predict what value it is
    out = cnn_model.predict(img)#predict the type

#Click to capture the image in the box
def captureImage(event,x,y,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("here")
        capture = params[0][height//3-(int)(height*.10)+2:height//3*2-(int)(height*.10), height//3+3:height//3*2]
        capture = cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY)
        return capture

def main():    
    cap = cv2.VideoCapture(0)
    #put text at (x,y) on given frame
    color = (0,0,255)
    scale = 2
    font = cv2.FONT_HERSHEY_PLAIN
    loadNN()#load the neural network
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




if __name__ == "__main__":
    main()
