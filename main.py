import cv2
import numpy as np
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D

NNPATH = "./checkpoint/checkpoint.ckpt"#path to NN


def loadNN():#load the model
    layers =  [Input(shape=(28,28,1),name='shape'),#make layers
        Conv2D(16,3,padding="same",activation="relu"),
        MaxPool2D(),
        Conv2D(32,3,padding="same",activation="relu"),
        MaxPool2D(),
        Flatten(),
        Dense(units=26, activation="softmax",name="out")]
    
    cnn_model = Sequential(layers)
    cnn_model.load_weights(NNPATH)#load the data
    cnn_model.summary()#print a summary of the model
    
    return cnn_model

cap = cv2.VideoCapture(0)

#put text at (x,y) on given frame
color = (0,0,255)
scale = 2
font = cv2.FONT_HERSHEY_PLAIN
def drawText(text, frame, x, y):
    cv2.putText(frame, text, (x, y), font, scale, color, 2)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't retrive frame")
        break
    cv2.imshow("Feed", frame)
    cv2.waitKey(1)
