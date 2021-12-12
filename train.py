import cv2 #do something with opencv
import numpy as np
import tensorflow as tf 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, MaxPool2D
from skimage.util import random_noise
import matplotlib
import matplotlib.pyplot as plt
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'#TODO remove this if you have a graphics card cuda can run on
#Ignore all the errors if you are not using the gpu to train
print(tf.__version__)
n_epochs = 3#how many times we train it

DATAPATH = "./data/"
CHECKPATH = "./newCheck/checkpoint.ckpt"#path to weight checkpoint

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPATH,
                                                 save_weights_only=True,
                                                 verbose=1)

def train(cnn_model,x_train,y_train,x_test,y_test,save):#this might take a bit, depending on if you are using GPU or not.
    print("\n---------------------------------------------------------------\nTRAINING")
    cnn_model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    if save:
        history = cnn_model.fit(x_train.reshape(-1, 28, 28 ,1), y_train, epochs=n_epochs,
                                validation_data=(x_test.reshape(-1, 28, 28 ,1), y_test),callbacks=[cp_callback])
    else:
        history = cnn_model.fit(x_train.reshape(-1, 28, 28 ,1), y_train, epochs=n_epochs,
                                validation_data=(x_test.reshape(-1, 28, 28 ,1), y_test))
    return history
    print("\n---------------------------------------------------------------\n")

def loadData():
    
    print("\n---------------------------------------------------------------\nLOADING DATA")

    labels = []
    pictures = []
    with open(DATAPATH+"train.csv", 'r') as f:
        lines = f.readlines()[1:]#skip label line
        i = 0
        for line in lines:
            vals = line.split(',')
            
            label = vals[0]#grab label
            labels.append(int(label))

            vals = vals[1:]
            pic = np.asarray(vals,dtype=np.uint8).reshape([28,28])#define the numpy array
            pic = cv2.adaptiveThreshold(pic,255,
                    cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,6)
            pic = cv2.morphologyEx(pic, cv2.MORPH_CLOSE, (3,3)) 
            pictures.append(pic)
            i += 1

    pictures = np.asarray(pictures)
    labels = np.asarray(labels)
    train = (pictures,labels)
    labels = []
    pictures = []
    with open(DATAPATH+"test.csv", 'r') as f:
        lines = f.readlines()[1:]#skip label line
        for line in lines:
            vals = line.split(',')
            
            label = vals[0]#grab label
            labels.append(int(label))

            vals = vals[1:]
            pic = np.asarray(vals,dtype=np.uint8).reshape([28,28])#define the numpy array
            pic = cv2.adaptiveThreshold(pic,255,
                    cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,6)
            pic = cv2.morphologyEx(pic, cv2.MORPH_CLOSE, (3,3)) 
            pictures.append(pic)
    pictures = np.asarray(pictures)
    labels = np.asarray(labels)

    test = (pictures,labels)
    print(f"Loaded Data. Train: {len(train[0])}, Test: {len(test[0])}\n---------------------------------------------------------------\n")
    return train,test

def stats(history):#plot stats
    print("\n---------------------------------------------------------------\nSTATS")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(1, n_epochs+1), history.history['loss'], label='Train set')
    plt.plot(np.arange(1, n_epochs+1), history.history['val_loss'], label='Test set')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, n_epochs+1), history.history['accuracy'], label='Train set')
    plt.plot(np.arange(1, n_epochs+1), history.history['val_accuracy'], label='Test set')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    print(f"\nAccuracy on the final epoch of training was {100*history.history['accuracy'][-1]:0.2f}%\n---------------------------------------------------------------\n")
    plt.show()
    
def makeModel():
    s = ""
    save = False
    while s == "":
        s = input("Save Model? Y/N: ").lower()
        if s == "y":
            save = True
        elif s == "n":
            save = False
        else:
            print(f"Unknown input: {s}")
            s = ""
 
   
    '''
    layers = [Input(shape=(28,28,1),name='shape'),#make layers

      Conv2D(32,3,padding="same",activation="relu"),
      MaxPool2D(),
      Conv2D(32,3,padding="same",activation="relu"),
      MaxPool2D(),
      Conv2D(64, 3, padding="same", activation="relu"),
      MaxPool2D(),
      Flatten(),
      Dense(units=256, activation="relu"),
      Dropout(.5),
      Dense(units=26, activation="softmax",name="out")]#note that we are including j and z, but they can not be seen in the data as they require movment
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
    

    model.summary()#print a summary of the model
    
    if len(os.listdir(os.path.dirname(CHECKPATH))) != 0:
        print("\n---------------------------------------------------------------\n")
        s = ""
        while s == "":
            s = input("Found model checkpoint, Load Model? Y/N: ").lower()
            if s == "y":
                print("Loading Model...")
                model.load_weights(CHECKPATH)
            elif s == "n":
                print("Not Loading")
            else:
                print(f"Unknown input: {s}")
                s = ""

    (x_train,y_train),(x_test,y_test) = loadData()#grab data

    

    history = train(model,x_train,y_train,x_test,y_test,save)
    
    stats(history)

if __name__ == "__main__":#run the main fun
    makeModel()
