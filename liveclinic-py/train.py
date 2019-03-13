# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import cv2

print(tf.__version__)

core_path = "/home/apoorv/Desktop/liveclinic/"
training_data_path = core_path + "/training_images/"
testing_data_path =  core_path + "/testing_images/"
training_data = []
testing_data = []
IMG_SIZE = 50
object = ['medicine strip']

for img in tqdm(os.listdir(training_data_path)):
    img = cv2.resize(cv2.imread(os.path.join(training_data_path,img),cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
    training_data.append([np.array(img),'0'])

for img in tqdm(os.listdir(testing_data_path)):
    img = cv2.resize(cv2.imread(os.path.join(testing_data_path,img),cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
    testing_data.append([np.array(img),'0'])
np.save('test_data.npy',testing_data)
