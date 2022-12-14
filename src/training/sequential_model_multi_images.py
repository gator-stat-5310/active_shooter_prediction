from collections import defaultdict
from cv2 import imread
from os import walk, chdir
from os.path import join
from tensorflow import keras
import fnmatch
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from os import walk, chdir, getcwd, makedirs
from os.path import join
from cv2 import imread, resize, imwrite, cvtColor, COLOR_BGR2GRAY, IMREAD_GRAYSCALE

import numpy as np
from PIL import Image
from pathlib import Path


def read_images(images_path):
    images = []
    labels = []
    for root, _, files in walk(images_path):
        for file in files:
            file_path = join(root, file)
            img_array = imread(file_path, IMREAD_GRAYSCALE)
            if img_array is not None:
                print(file_path)
                img_pil = Image.fromarray(img_array)
                img_28x28 = np.array(img_pil.resize((28, 28), Image.ANTIALIAS))
                img_array = (img_28x28.flatten())
                # img_array  = img_array.reshape(-1,1).T
                images.append(img_array)
                labels.append(Path(images_path).stem)
    return images, labels

root_path = r'C:/Users/JasSu/Documents/UHD/Sem7/capstone_project/active_shooter_prediction/presentation/data'
data_dir_paths = ["assault_rifles","m16", "machine_guns",  "pistols", "revolvers","rifles","shot_guns",
                  "sub_machine_guns"]

X_train = []
y_train = []
for data_path in data_dir_paths:
    full_data_path = join(root_path, data_path)
    X_training, y_training = read_images(full_data_path)
    X_train.append(X_training)
    y_train.append(y_training)


X_train_ = np.concatenate((np.array(X_train[0]), np.array(X_train[1])))
for i in range(2,8):
    X_train_ = np.concatenate((X_train_, np.array(X_train[i])))

y_train_ = np.concatenate((np.array(y_train[0]), np.array(y_train[1])))
for i in range(2,8):
    y_train_ = np.concatenate((y_train_, np.array(y_train[i])))

X_train = np.array(X_train_)/255
y_train = np.array(y_train_)
#print(X_train)
print(X_train.shape)
y_train_labels = np.array(y_train)
print(y_train.shape)
print(y_train[1:10])

keys = {"assault_rifles":0,"m16":1, "machine_guns":2,  "pistols":3, "revolvers":4,"rifles":5,"shot_guns":6,
        "sub_machine_guns":7}

y_train_labels=np.ones(y_train.shape[0], dtype=int)
for index in range(0, y_train.shape[0]):
    y_train_labels[index] = keys[y_train[index]]

print(y_train_labels.shape)


# build a simple sequential neural network
model2 = keras.models.Sequential()
model2.add(keras.Input(shape=(784,)))
#model2.add(keras.layers.Flatten(input_shape=(28,28)))

model2.add(keras.layers.Dense(100, activation='relu'))

# Dense layer means that all the inputs neurons are connected with the output neurons
# The following nerual network is a stacked neural network with 10 outputs and an input of 10
model2.add(keras.layers.Dense(8, activation='sigmoid'))

#optimizer helps efficientily learn using back propagation
                          #loss use the entropy value for difference between
model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model2.fit(X_train, y_train_labels, epochs=20)
# the accuracy jumped to 98% simply by scaling or normalizing the data

X_test = X_train[0:50]
y_test_labels = y_train_labels[0:50]
model2.evaluate(X_test, y_test_labels)

y_test_predicted = model2.predict(X_test)
y_predicted_labels = [np.argmax(y_predicted) for y_predicted in y_test_predicted ]

conf_matrix = tf.math.confusion_matrix(labels=y_test_labels, predictions=y_predicted_labels)

print(conf_matrix)
import seaborn as sn
plt.figure(figsize=(10,7))

# generate heap map of confusion matrix
sn.heatmap(conf_matrix, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

input("---end---")