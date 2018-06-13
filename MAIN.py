# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 23:18:00 2018

@author: BI 2018
"""

#%%
from PIL import Image
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import numpy 
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# Create a sequential model
mod = Sequential()

# Add network's level
mod.add(Dense(900, input_dim=784, activation="relu", kernel_initializer="normal"))
mod.add(Dense(10, activation="softmax", kernel_initializer="normal"))

# Compile model
mod.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

# Teach the network
mod.fit(X_train, Y_train, batch_size=300, epochs=10, validation_split=0.2, verbose=2)

# Test the quality of network 
accuracy = mod.evaluate(X_test, Y_test, verbose=0)
print ("Accuracy of Network :" , accuracy[1]*100 , "%")
im = Image.open('4.png')
im_grey = im.convert('L')
im_array = numpy.array(im_grey)
im_array=numpy.reshape(im_array, (1, 784)).astype('float32')
x = 255 - im_array
x /= 255
plt.imshow(im, cmap='gray')
plt.show
print("Result of prediction :", numpy.argmax(mod.predict(x), axis=1))