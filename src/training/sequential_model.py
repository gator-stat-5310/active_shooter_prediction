from collections import defaultdict
from cv2 import imread
from os import walk, chdir
from os.path import join
from tensorflow import keras
import fnmatch
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def get_files(dir_path) -> tuple:
    file_paths=[]
    labels=[]
    for root, _, files in walk(dir_path):
        for file in fnmatch.filter(files, "*.jpg"):
            file_path = join(root, file)
            file_paths.append(file_path)
            labels.append(file)
    return file_paths, labels


def load_image(image_path):
    img = imread(image_path)
    return img


def load_images(image_dir):
    images =defaultdict()
    files, labels = get_files(image_dir)
    for file, label in zip(files, labels):
        img = imread(file)
        images[label]= img
    return images

from os import walk, chdir, getcwd, makedirs
from os.path import join
from cv2 import imread, resize, imwrite, cvtColor, COLOR_BGR2GRAY, IMREAD_GRAYSCALE

import numpy as np
from PIL import Image
from pathlib import Path

def read_images(images_path):
    images=[]
    labels=[]
    for root, _, files in walk(images_path):
        for file in files:
            file_path = join(root, file)
            img_array = imread(file_path, IMREAD_GRAYSCALE)
            if img_array is not None:
                print(file_path)
                img_pil = Image.fromarray(img_array)
                img_28x28 = np.array(img_pil.resize((28, 28), Image.ANTIALIAS))
                img_array = (img_28x28.flatten())
                #img_array  = img_array.reshape(-1,1).T
                images.append(img_array)
                labels.append(Path(file).stem)
    return images, labels



data_dir_paths = ["C:/Users/JasSu/Documents/UHD/Sem7/capstone_project/active_shooter_prediction/presentation/data/m16",
                  "C:/Users/JasSu/Documents/UHD/Sem7/capstone_project/active_shooter_prediction/presentation/data/rifles",
                  "C:/Users/JasSu/Documents/UHD/Sem7/capstone_project/active_shooter_prediction/presentation/data/pistols"]

for data_path in data_dir_paths:
    X_train, y_train = read_images(data_path)

X_train = np.array(X_train)


# build a simple sequential neural network
model2 = keras.models.Sequential()
model2.add(keras.Input(shape=(784,)))
#model2.add(keras.layers.Flatten(input_shape=(28,28)))

#model2.add(keras.layers.Dense(100, activation='relu'))

# Dense layer means that all the inputs neurons are connected with the output neurons
# The following nerual network is a stacked neural network with 10 outputs and an input of 10
model2.add(keras.layers.Dense(111, activation='sigmoid'))

#optimizer helps efficientily learn using back propagation
                          #loss use the entropy value for difference between
model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model2.fit(X_train, y_train, epochs=5)
# the accuracy jumped to 98% simply by scaling or normalizing the data

# model2.evaluate(X_test, y_test)
#
# y_test_predicted = model2.predict(X_test)
# y_predicted_labels = [np.argmax(y_predicted) for y_predicted in y_test_predicted ]
#
# conf_matrix = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
#
# import seaborn as sn
# plt.figure(figsize=(10,7))
#
# # generate heap map of confusion matrix
# sn.heatmap(conf_matrix, annot=True, fmt='d')
# plt.xlabel('Predicted')
# plt.ylabel('Truth')