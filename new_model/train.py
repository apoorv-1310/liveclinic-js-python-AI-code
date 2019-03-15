import os
import sys
import keras
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import keras.backend as K
import tensorflowjs as tfjs
from keras import optimizers
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Sequential
from matplotlib.pyplot import imshow
from keras.preprocessing import image
from keras.layers import Activation, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D

img_width, img_height = 150, 150
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
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam', metrics=['accuracy'])

def startTraining():
    train_data_dir = '/home/apoorv/Desktop/apps/liveclinic/new_model/training_data'
    validation_data_dir = '/home/apoorv/Desktop/apps/liveclinic/new_model/validation_data'
    train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=30,width_shift_range=0.2,shear_range=0.2,zoom_range=0.5,horizontal_flip=True,vertical_flip=True)
    validation_datagen = ImageDataGenerator(rescale=1./255,rotation_range=30,width_shift_range=0.2,shear_range=0.2,zoom_range=0.5,horizontal_flip=True,vertical_flip=True)
    train_generator = train_datagen.flow_from_directory(train_data_dir,target_size=(img_width, img_height),batch_size=6,class_mode='binary')
    validation_generator = validation_datagen.flow_from_directory(validation_data_dir,target_size=(img_width, img_height),batch_size=3,class_mode='binary')
    nb_epoch = 50
    nb_train_samples = 10
    nb_validation_samples = 10

    # model.load_weights('working_model.h5')
    model.fit_generator(train_generator,samples_per_epoch=nb_train_samples,epochs=nb_epoch,validation_data=validation_generator,validation_steps=nb_validation_samples)

    print ("Loaded")
    model.summary()
    model.save("working_model.h5")

    print("ALL GOOD TILL NOW")

def startPredicting():
    model.load_weights('working_model.h5')
    classes = ["medicine strip","pill bottle",'pill']

    dir_path = os.path.dirname(os.path.realpath(__file__))
    medicine_strip_testing_data_path = dir_path+"/testing_data/medicine_strip"
    pill_bottle_testing_path = dir_path + "/testing_data/pill_bottle/"

    print("------------------------------")
    print("SCANNING MEDICINE STRIP FOLDER")
    print("------------------------------")

    for img in os.listdir(medicine_strip_testing_data_path):
        image_full_path = medicine_strip_testing_data_path+"/"+img
        print("------------------------------")
        read_img = mpimg.imread(image_full_path)
        imgplot = plt.imshow(read_img)
        plt.show()
        img1 = image.load_img(image_full_path, target_size=(img_width,img_height))
        x = image.img_to_array(img1)
        x = np.expand_dims(x,axis=0)
        x /=255
        predicted_classes = model.predict_classes(x)
        print("THAT WAS ======>",classes[predicted_classes[0]])

    print("------------------------------")
    print("SCANNING PILL BOTTLE FOLDER")
    print("------------------------------")

    for img in os.listdir(pill_bottle_testing_path):
        image_full_path = pill_bottle_testing_path+"/"+img
        print("------------------------------")
        read_img = mpimg.imread(image_full_path)
        imgplot = plt.imshow(read_img)
        plt.show()
        img1 = image.load_img(image_full_path, target_size=(img_width,img_height))
        x = image.img_to_array(img1)
        x = np.expand_dims(x,axis=0)
        x /=255
        predicted_classes = model.predict_classes(x)
        print("THAT WAS ======>",classes[predicted_classes[0]])

def convert():
    print("Converting...")
    model.load_weights('working_model.h5')
    tfjs.converters.save_keras_model(model, "tfjsmodel.json")

if(len(sys.argv) > 1 ):
    program_name = sys.argv[1]
    if(program_name == "train"):
        startTraining()
    if(program_name == "predict"):
        startPredicting()
    if(program_name == "convert"):
        convert()
else:
    print ("PLEASE SEND TASK (train or predict or convert) for example python train.py train")
