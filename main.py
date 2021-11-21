import cv2 #do something with opencv
import numpy as np
import tensorflow as tf 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D
import matplotlib
import matplotlib.pyplot as plt
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'#TODO remove this if you have a graphics card cuda can run on
#Ignore all the errors if you are not using the gpu to train

n_epochs = 3#how many times we train it

PATH = "./data"

def train():
    cnn_model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = cnn_model.fit(x_train.reshape(-1, 28, 28 ,1), y_train, epochs=n_epochs,
                            validation_data=(x_test.reshape(-1, 28, 28 ,1), y_test))


def loadData():


def stats(history):#plot stats
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
    print(f"\nAccuracy on the final epoch of training was {100*history.history['accuracy'][-1]:0.2f}%")

def main():
    train,test = loadData();

    cnn_layers =  [Input(shape=(28,28,1),name='shape'),#make layers
      Conv2D(16,3,padding="same",activation="relu"),
      MaxPool2D(),
      Conv2D(32,3,padding="same",activation="relu"),
      MaxPool2D(),
      Flatten(),
      Dense(units=26, activation="softmax",name="out")]#note that we are including j and z, but they can not be seen in the data as they require movment
    
    cnn_model = Sequential(cnn_layers)
    cnn_model.summary()
    



if __name__ == "__main__":#run the main fun
    main()
