import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
import matplotlib.pyplot as plt
import keras.backend as K
from matplotlib.pyplot import imshow
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Activation, Dense
img_width, img_height = 150, 150

def startTraining():
    train_data_dir = 'training_data'
    validation_data_dir = 'validation_data'
    train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=30,width_shift_range=0.2,shear_range=0.2,zoom_range=0.5,horizontal_flip=True,vertical_flip=True)
    validation_datagen = ImageDataGenerator(rescale=1./255,rotation_range=30,width_shift_range=0.2,shear_range=0.2,zoom_range=0.5,horizontal_flip=True,vertical_flip=True)
    train_generator = train_datagen.flow_from_directory(train_data_dir,target_size=(img_width, img_height),batch_size=6,class_mode='binary')
    validation_generator = validation_datagen.flow_from_directory(validation_data_dir,target_size=(img_width, img_height),batch_size=3,class_mode='binary')

    # Small conv net
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), input_shape=(img_width, img_height,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(70))
    model.add(Activation('sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

    nb_epoch = 200
    nb_train_samples = 10
    nb_validation_samples = 10

    model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples)

    print ("Loaded")
    model.summary()
    model.save("working_model.h5")
    model.load_weights('working_model.h5')

    print("ALL GOOD TILL NOW")

startTraining()
