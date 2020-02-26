"""
Created on Tue Mar  9 18:51:38 2019
@author: Milad
"""

from keras.preprocessing import image
from keras.applications.xception import Xception 
from keras.utils import to_categorical
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, MaxPooling2D, Conv2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K

from keras.callbacks import ModelCheckpoint,LearningRateScheduler,CSVLogger
import keras
import cv2

import numpy as np
import csv
import pickle

K.tensorflow_backend._get_available_gpus()

with open('test_images.pkl', 'rb') as f:
    raw_data = pickle.load(f)
 
row = 71
col = 71

data = []
for img in raw_data:
    img = (cv2.resize(img,(int(col),int(row)))).astype('float32')
    cvuint8 = cv2.convertScaleAbs(img)
    data.append(cv2.cvtColor(cvuint8, cv2.COLOR_GRAY2RGB))

x_test = np.array(data)/255 

input_shape = (row,row,3)

model = Xception(include_top=True, weights='AA_weights.hdf5', input_tensor=None, input_shape=input_shape, pooling=None, classes = 10)
      
Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

result_class = model.predict(x_test) 
result_class = np.array(np.argmax(result_class, axis=1), 'int').reshape(x_test.shape[0],1)
result_id = np.arange(0,x_test.shape[0]).reshape(-1,1)

result = np.concatenate((result_id,  result_class), axis=1)
np.savetxt("xception.csv", result, fmt='%i', header='Id,Category', delimiter=',', comments='')








