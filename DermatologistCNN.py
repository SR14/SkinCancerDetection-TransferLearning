#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 18:11:08 2019

@author: sergiorobledo
"""
from random import seed
'''               '''
''' Get Filenames '''
'''               '''

data_path = '/Users/sergiorobledo/Desktop/Dermatologist AI/data'

import glob
train_mf = glob.glob(data_path + '/train/melanoma/resized/*.jpg')
train_nf = glob.glob(data_path + '/train/nevus/resized/*.jpg')
train_sf = glob.glob(data_path + '/train/seborrheic_keratosis/resized/*.jpg')

valid_mf = glob.glob(data_path + '/valid/melanoma/resized/*.jpg')
valid_nf = glob.glob(data_path + '/valid/nevus/resized/*.jpg')
valid_sf = glob.glob(data_path + '/valid/seborrheic_keratosis/resized/*.jpg')

test_mf = glob.glob(data_path + '/test/melanoma/resized/*.jpg')
test_nf = glob.glob(data_path + '/test/nevus/resized/*.jpg')
test_sf = glob.glob(data_path + '/test/seborrheic_keratosis/resized/*.jpg')

'''                        '''
''' Load Data & Preprocess '''
'''                        '''

from PIL import Image
import numpy as np

train_mel = np.array([np.array(Image.open(item)) for item in train_mf])
train_nev = np.array([np.array(Image.open(item)) for item in train_nf])
train_seb = np.array([np.array(Image.open(item)) for item in train_sf])
# stack images
x_train = np.concatenate((train_mel,train_nev,train_seb))
# rescale data 
x_train = x_train.astype('float32')/255

valid_mel = np.array([np.array(Image.open(item)) for item in valid_mf])
valid_nev = np.array([np.array(Image.open(item)) for item in valid_nf])
valid_seb = np.array([np.array(Image.open(item)) for item in valid_sf])
x_valid = np.concatenate((valid_mel, valid_nev, valid_seb))
x_valid = x_valid.astype('float32')/255

test_mel = np.array([np.array(Image.open(item)) for item in test_mf])
test_nev = np.array([np.array(Image.open(item)) for item in test_nf])
test_seb = np.array([np.array(Image.open(item)) for item in test_sf])
x_test = np.concatenate((test_mel, test_nev, test_seb))
x_test = x_test.astype('float32')/255

''' Create Label Data '''    
   
def label_creation(mel_size, nev_size, seb_size, label_arr):
    for i in range(mel_size):
        label_arr.append(0.0)
    for i in range(nev_size):
        label_arr.append(1.0)
    for i in range(seb_size):
        label_arr.append(2.0)

y_train = []
label_creation(374, 1372, 254, y_train)
y_train = np.array(y_train)

y_valid = []
label_creation(30, 78, 42, y_valid)
y_valid = np.array(y_valid)

y_test = []
label_creation(117, 393, 90, y_test)
y_test = np.array(y_test)

''' One Hot Encode Labels ''' 

from keras.utils import to_categorical
y_train = to_categorical(y_train,3)
y_valid = to_categorical(y_valid, 3)
y_test_one = to_categorical(y_test, 3)

'''                   '''
''' Data Augmentation '''
'''                   '''

from keras.preprocessing.image import ImageDataGenerator
seed(714)
datagen_train = ImageDataGenerator(width_shift_range = 0.1,
                                   height_shift_range = 0.1,
                                   horizontal_flip = True)
seed(714)
datagen_valid = ImageDataGenerator(width_shift_range = 0.1,
                                   height_shift_range = 0.1,
                                   horizontal_flip = True)
seed(714)
datagen_train.fit(x_train)
datagen_valid.fit(x_valid)

'''                    '''
''' Model Architecture '''
'''                    '''

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout

model = Sequential()
model.add(Conv2D(filters = 32, 
                 kernel_size = 3,
                 activation = 'relu',
                 input_shape=(256, 256, 3)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 32, 
                 kernel_size = 3,
                 activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))

model.add(Conv2D(filters = 64, 
                 kernel_size = 3,
                 activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))

model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation = 'softmax'))

''' Model Compile ''' 

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['accuracy'])

''' Fit Model '''

from keras.callbacks import ModelCheckpoint   

batch_size = 16
epochs = 50

# train the model
checkpointer = ModelCheckpoint(filepath='derma.weights.best.hdf5', verbose=1, 
                               save_best_only=True)
seed(714)
model.fit_generator(datagen_train.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch= 2000 // batch_size,
                    epochs=epochs, verbose=2, callbacks=[checkpointer],
                    validation_data=datagen_valid.flow(x_valid, y_valid, batch_size=batch_size),
                    validation_steps= 800 // batch_size)

'''                     '''
'''  Best Weights Model '''
'''                     '''

''' Load Weights ''' 

model.load_weights('derma.weights.best.hdf5')

''' Test Set Accuracy '''

seed(714)
score = model.evaluate(x_test, y_test_one, verbose = 0)
print('\n', 'Test accuracy:', score[1])


''' Get Predictions '''

seed(714)
cnn_pred = model.predict(x_test, verbose = 1)

'''                        '''
''' Output Predictions CSV '''
'''                        '''

''' Melanoma & SK Predicted Probabilities ''' 
csv_pred = cnn_pred[:,(0,2)]

''' Create Name List ''' 

image_arr = []

def image_name(filelist):
    for i in range(len(filelist)):
        image_arr.append(filelist[i][50:])

image_name(test_mf)
image_name(test_nf)
image_name(test_sf)

''' Join Image Name and Predicted Probabilities ''' 

csv_full = []
for i in range(len(image_arr)):
    csv_full = [image_arr[i] + ',' + str(pred[0]) + ',' + str(pred[1]) for pred in csv_pred]

''' Write CSV '''

import csv
with open('predictions.csv', 'w') as predictions:
    prediction_writer = csv.writer(predictions, 
                                   delimiter=',', 
                                   quotechar = " ",
                                   quoting = csv.QUOTE_MINIMAL)
    prediction_writer.writerow(['Id', 'task_1', 'task_2'])  
    for i in range(len(csv_full)):
        prediction_writer.writerow([csv_full[i]])






















































































































