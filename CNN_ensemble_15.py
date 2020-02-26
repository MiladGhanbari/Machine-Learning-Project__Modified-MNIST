# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 20:51:38 2019
CNNs on MNIST database
@author: Milad
"""

Data_Path = "F:\McGill\Courses\Machine Learning\Projects\Third Project\Data\input"

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import svm



## Loading image data
print(os.listdir(Data_Path))

train_images = pd.read_pickle(Data_Path + '/train_images.pkl')
train_labels = pd.read_csv(Data_Path + '/train_labels.csv')
test_images = pd.read_pickle(Data_Path + '/test_images.pkl')

Ytrain = np.ndarray(shape=(train_images.shape[0] ,1), dtype=int)
for i in range(train_images.shape[0]):
    Ytrain[i] = train_labels.iloc[i]['Category']


############### PreProcessing function ###################### 
Img_size = 28
def PreProcessing(ImageMatrix):
    X_out = np.ndarray(shape=(ImageMatrix.shape[0] ,Img_size, Img_size), dtype=np.uint8)
    
    X = np.ndarray(shape=(ImageMatrix.shape[0] ,64, 64), dtype=np.uint8)
    for i in range(ImageMatrix.shape[0]):
        X[i][:][:] = ImageMatrix[i]
    from PIL import Image
    for ind in range(ImageMatrix.shape[0]):
        import cv2
        image =  X[ind]
        image_white = image
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if(image[i][j] < 240):
                    image_white[i][j] = 0
        Filt_img = image_white        

        ## filtering 
        kernel = np.ones((1,1),np.uint8)
        img_opening = cv2.morphologyEx(image_white, cv2.MORPH_OPEN, kernel)
        img_closing = cv2.morphologyEx(img_opening, cv2.MORPH_CLOSE, kernel)
        
        scale = 1
        Large_img = cv2.resize(img_closing, (0,0), fx=scale, fy=scale) 
        Large_img_BW = cv2.resize(image_white, (0,0), fx=scale, fy=scale) 

        
        im2, contours, hierarchy = cv2.findContours(Large_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        Ws = [] 
        Hs = []
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            Ws.append(w)
            Hs.append(h)
        max_h = max(Hs)
        max_w = max(Ws)
        max_l = max(max_h,max_w)
    
        # Find the index of the contour with largest width
        if max_l == max_h:
            max_index = np.argmax(Hs)
        else:
            max_index = np.argmax(Ws)
                       
        rect = cv2.minAreaRect(contours[max_index])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        mask = np.zeros_like(Large_img) # Create mask where white is what we want, black otherwise
        cv2.drawContours(mask, [box], 0, (255, 255, 255), -1) # Draw filled contour in mask
        
        out = np.zeros_like(Large_img_BW) # Extract out the object and place into output image
        out[mask == 255] = Large_img_BW[mask == 255]
        
        #find center of the component
        x = rect[0][0]
        y = rect[0][1]
        px = x+max_l/2
        py = y+max_l/2        
    
        preprocessed_img = np.zeros(X[0].shape)
        preprocessed_img = Large_img_BW*out
    
    
        # To shift the largest component to the center of an image
        preprocessed_img = preprocessed_img.astype('uint8')
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(preprocessed_img, connectivity=4)
        sizes = stats[:, -1]
        max_label = 1
        max_size = sizes[1]
        for i in range(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]
            
        img2 = np.zeros(output.shape)
        img2[output == max_label] = 255.
        
        
        # To shift the largest component to the center of an image
        px = 32*scale - centroids[max_label][0]
        py = 32*scale - centroids[max_label][1]
        M = np.float32([[1,0,px],[0,1,py]])
        dst = cv2.warpAffine(img2,M,(64*scale,64*scale))
        
        small_img = cv2.resize(dst, (0,0), fx=1/scale, fy=1/scale) 
        
        small_img = small_img[18:46,18:46]    

        X_out[ind][:][:] = small_img
        
    return X_out
 

############### PreProcessing data ######################     
xtrain = PreProcessing(train_images)
xtest = PreProcessing(test_images)
ytrain = Ytrain


###################  Ensembling Multiple CNNs #######################
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


# BUILD CONVOLUTIONAL NEURAL NETWORKS
nets = 15
model = [0] *nets
for j in range(nets):
    model[j] = Sequential()
    model[j].add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(1, Img_size,Img_size)))
    model[j].add(LeakyReLU(alpha=0.1))
    model[j].add(MaxPooling2D((2, 2),padding='same'))
    model[j].add(Dropout(0.25))
    model[j].add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    model[j].add(LeakyReLU(alpha=0.1))
    model[j].add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model[j].add(Dropout(0.25))
    model[j].add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    model[j].add(LeakyReLU(alpha=0.1))                  
    model[j].add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model[j].add(Dropout(0.4))
    model[j].add(Flatten())
    model[j].add(Dense(128, activation='linear'))
    model[j].add(LeakyReLU(alpha=0.1))           
    model[j].add(Dropout(0.3))
    model[j].add(Dense(10, activation='softmax'))
    model[j].compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])


# TRAIN NETWORKS
history = [0] * nets
accTr = 0
accVal = 0

from keras.callbacks import LearningRateScheduler
# learning rate schedule
def step_decay(epoch):
	initial_lrate = 0.001
	lrate = initial_lrate * 0.995 ** epoch
	return lrate
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

for j in range(nets):
    x_train, x_val, y_train, y_val1 = train_test_split(xtrain, ytrain, train_size = 0.9, test_size = 0.1) 
    x_train = x_train.reshape(-1, 1, Img_size,Img_size)
    x_val = x_val.reshape(-1, 1, Img_size,Img_size)
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_train = x_train / 255.
    x_val = x_val / 255.
    num_category = 10
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_category)
    y_val = keras.utils.to_categorical(y_val1, num_category)

    history[j] = model[j].fit(x_train, y_train, batch_size=150,epochs=50 ,verbose=1, callbacks=callbacks_list, validation_data=(x_val, y_val))
    
    EvTr = model[j].evaluate(x_train, y_train, verbose=0)
    accTr += EvTr[1]
    EvVal = model[j].evaluate(x_val, y_val, verbose=0)
    accVal += EvVal[1]



# Accuracy
print('CNN Training Accuracy: {}'.format(accTr/nets))
print('CNN Validation Accuracy: {}' .format(accVal/nets))


