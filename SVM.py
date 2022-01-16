# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 20:51:38 2019
Support Vector Machines on MNIST database
@author: Milad
"""

Data_Path = "" # the path to data

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
ytrain = Ytrain


############### SVM Classifier ##########################
# spliting training data 
x_train_svm, x_val_svm, y_train_svm, y_val_svm = train_test_split(xtrain, ytrain, train_size = 0.9, test_size = 0.1) 
# Convert matrix format to vector format for regression
x_train_svmf = np.zeros((x_train_svm.shape[0], x_train_svm.shape[1] * x_train_svm.shape[2]))
for i in range(x_train_svm.shape[0]):
    x_train_lrf[i][:] = np.array(x_train_svm[i].reshape(-1))/255
x_val_svmf = np.zeros((x_val_svm.shape[0], x_val_svm.shape[1] * x_val_svm.shape[2]))
for i in range(x_val_svm.shape[0]):
    x_val_svmf[i][:] = np.array(x_val_svm[i].reshape(-1))/255
# Model initialization
classifier = svm.LinearSVC()
# Fit the data(train the model)
classifier.fit(x_train_lrf, y_train_svm)
# Accuracy
print('SVM Training Accuracy: {}'.format(classifier.score(x_train_lrf, y_train_svm)))
print('SVM  Validation Accuracy: {}' .format(classifier.score(x_val_svmf, y_val_svm)))


    


