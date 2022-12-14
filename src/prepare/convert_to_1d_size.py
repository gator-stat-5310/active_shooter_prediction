from cv2 import imread
import numpy as np
from PIL import Image
import cv2
import os




image_path = "/presentation/data_resize/280px-STM-556.jpg"
img_array = imread(image_path, cv2.IMREAD_GRAYSCALE)

img_pil = Image.fromarray(img_array)
img_28x28 = np.array(img_pil.resize((28, 28), Image.ANTIALIAS))

img_array = (img_28x28.flatten())

img_array  = img_array.reshape(-1,1).T

print(img_array)
