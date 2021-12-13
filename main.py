import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import os


ALPHA = "abcdefghijklmnopqrstuvwxyz"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'#TODO remove this if you have a graphics card cuda can run on
#Ignore all the errors if you are not using the gpu to train

NNPATH = "./newCheck/checkpoint.ckpt"#path to NN

toDraw = ""

def drawText(text, frame, x, y):
    image = cv2.putText(frame, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,100), 2)

def loadNN():#load the model
    '''
    layers =  [Input(shape=(28,28,1),name='shape'),#make layers
      Conv2D(32,3,padding="same",activation="relu"),
      MaxPool2D(),
      Conv2D(32,3,padding="same",activation="relu"),
      MaxPool2D(),
      Conv2D(64, 3, padding="same", activation="relu"),
      MaxPool2D(),
      Flatten(),
      Dense(units=256, activation="relu"),
      Dropout(.5),
      Dense(units=26, activation="softmax",name="out")]
    '''
    
    model = Sequential()

    model.add(Input(shape=(28,28,1),name="Input"))
    model.add(Conv2D(filters=16,kernel_size=(2,2),padding="same", activation="relu"))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=64, kernel_size=(5,5), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(5,5), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(5,5), padding="same", activation="relu"))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=26, activation="softmax"))

    model.load_weights(NNPATH)#load the data
    model.summary()#print a summary of the model
    return model

def predict(img,model):#predict what value it is
    img = cv2.adaptiveThreshold(img,255,
            cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,5,6)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, (3,3))

    cv2.imshow("INPUT",img)
    out = model.predict(img.reshape(-1,28,28))[0]#predict the type
    return np.argmax(out)
#Click to capture the image in the box
def captureImage(event,x,y,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
        global toDraw
        capture = params[0][params[1]//3-(int)(params[1]*.10)+2:params[1]//3*2-(int)(params[1]*.10), params[1]//3+3:params[1]//3*2]
        capture = cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY)
        capture = cv2.resize(capture, (28,28), interpolation = cv2.INTER_AREA)
        letter = ALPHA[predict(capture,params[2])]
        print(letter)
        toDraw += letter

        if len(toDraw) > 3:
            toDraw = letter
        

def main():
    print("starting")
    cap = cv2.VideoCapture(0)
    #put text at (x,y) on given frame
    color = (0,0,255)
    scale = 2
    font = cv2.FONT_HERSHEY_PLAIN
    cnn_model = loadNN()#load the neural network
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
        
        drawText(toDraw,frame, 100, 100)#draw the text buffer

        cv2.namedWindow("Feed")
        cv2.imshow("Feed", frame)
        params = [frame,height,cnn_model]
        cv2.setMouseCallback("Feed",captureImage, params)
        
        cv2.waitKey(1)



if __name__ == "__main__":
    main()
