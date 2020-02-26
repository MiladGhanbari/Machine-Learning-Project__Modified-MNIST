"""
Created on Tue Mar  8 21:10:38 2019
@author: Milad
"""

import math
from keras.preprocessing import image
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
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


with open('train_images.pkl', 'rb') as f:
    raw_data = pickle.load(f)

lables = []
with open('train_labels.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for i, row in enumerate(reader):
        if i > 0:
            lables.append(int(row[1]))

row = 71
col = 71
input_shape = (row,col,3)

data = []
for img in raw_data:
    img = (cv2.resize(img,(int(col),int(row)))).astype('float32')
    cvuint8 = cv2.convertScaleAbs(img)
    data.append(cv2.cvtColor(cvuint8, cv2.COLOR_GRAY2RGB))

score = []
for i in range(20):

    print(i)
    if i == 0:
        x_train = np.array(data[:38000])/255
    elif i == 19:
        x_train = np.array(data[2000:])/255
    else:
        x_train = np.concatenate((np.array(data[:38000-(2000*i)]), np.array(data[40000-(2000*i):])), axis=0)/ 255
    
    x_valid = np.array(data[38000-(2000*i):40000-(2000*i)])/ 255
    print(x_train.shape,x_valid.shape)

    if i == 0:
        y_train = lables[:38000]
    elif i == 9:
        y_train = lables[2000:]
    else:
        y_train = np.concatenate((lables[:38000-(2000*i)], lables[40000-(2000*i):]), axis=0)
    
    y_train = to_categorical(y_train, 10)
    y_valid = lables[38000-(2000*i):40000-(2000*i)]
    y_valid = to_categorical(y_valid, 10)

    print(y_train.shape, y_valid.shape)

    x_train = x_train.reshape(x_train.shape[0], row, row, 3)
    x_valid = x_valid.reshape(x_valid.shape[0], row, row, 3)

    y_train = y_train.reshape(y_train.shape[0], 10)
    y_valid = y_valid.reshape(y_valid.shape[0], 10)


    datagen = keras.preprocessing.image.ImageDataGenerator(featurewise_center=False, samplewise_center=False, 
                                             featurewise_std_normalization=False, samplewise_std_normalization=False, 
                                             zca_whitening=False, zca_epsilon=1e-06, rotation_range=45, width_shift_range=0.08, 
                                             height_shift_range=0.08, shear_range=0, zoom_range=0, 
                                             channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=False, 
                                             vertical_flip=False, rescale=1, preprocessing_function=None, data_format=None)


    datagen_test = keras.preprocessing.image.ImageDataGenerator(featurewise_center=False, samplewise_center=False,
                                             featurewise_std_normalization=False, samplewise_std_normalization=False,
                                             zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0,
                                             height_shift_range=0, shear_range=0, zoom_range=0,
                                             channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=False,
                                             vertical_flip=False, rescale=1, preprocessing_function=None, data_format=None)


    traning_set = datagen.flow(x_train, y_train, batch_size=50)
    validation_set = datagen_test.flow(x_valid, y_valid, batch_size=50)



    model = Xception(include_top=True, weights=None, input_tensor=None, input_shape=input_shape, pooling=None, classes = 10)


    def step_decay(epoch):
        
        initial_lrate = 0.005
        drop = 0.94
        epochs_drop = 2
        
        if epoch%epochs_drop == 0:
            lrate = initial_lrate * (drop ** np.floor(epoch/epochs_drop)) 
        else:
            lrate = initial_lrate * (drop ** np.floor((epoch-1)/epochs_drop)) 

        print(lrate)
        return lrate

    lrate = LearningRateScheduler(step_decay)

    Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


    model.compile(optimizer='Adam',

              loss='categorical_crossentropy', 

              metrics=['accuracy'])


    csv_logger = CSVLogger('./training_Xception_ten_res{}.log'.format(i), append=True)
    checkpointer = ModelCheckpoint(filepath='./weights_Xception_ten_res{}.hdf5'.format(i), verbose=1, save_best_only=True, monitor = 'val_acc')
    callbacks_list = [checkpointer,csv_logger, lrate]
    model.fit_generator(traning_set, epochs=50, steps_per_epoch=len(x_train) / 50, verbose=1, validation_data=validation_set, validation_steps=len(x_valid)/50, callbacks=callbacks_list)

    score.append(model.evaluate(x_valid, y_valid, verbose=0))
    print(score[i])








