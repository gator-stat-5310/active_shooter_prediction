from os import walk, chdir, getcwd, makedirs
from os.path import join
from cv2 import imread, resize, imwrite, cvtColor, COLOR_BGR2GRAY, IMREAD_GRAYSCALE
from pathlib import Path

import numpy as np
from PIL import Image
from pathlib import Path

# ("C:/Users/JasSu/Documents/UHD/Sem7/capstone_project/active_shooter_prediction/presentation")
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
                img_array  = img_array.reshape(-1,1).T
                images.append(img_array)
                labels.append(Path(file).stem)
    return images, labels


def resize_images(images_path, output_path):
    makedirs(output_path, exist_ok=True)

    for root, _, files in walk(images_path):
        for file in files:
            file_path = join(root, file)
            img = imread(file_path)
            if img is not None:
                grayImage = cvtColor(img, COLOR_BGR2GRAY)
                img_new = resize(grayImage, (640, 640))
                file_path = join(output_path, file)
                imwrite(file_path, img_new)


def resize_image(image_path, output_path):
    import pathlib
    img = imread(image_path)
    if img is not None:
        grayImage = cvtColor(img, COLOR_BGR2GRAY)
        img_new = resize(grayImage, (640, 640))
        image_path_as_path = pathlib.Path(image_path)
        file_path = join(output_path, f"{image_path_as_path.stem}_640.jpg")
        imwrite(file_path, img_new)


def process_image():
    source_path = r"C:\Users\JasSu\Documents\UHD\Sem7\capstone_project\active_shooter_prediction\yolov5\data_merged"
    target_path = r"C:\Users\JasSu\Documents\UHD\Sem7\capstone_project\active_shooter_prediction\yolov5\data_resized"
    image_dirs = ["assault_rifles", "machine_guns", "pistols", "shot_guns"]
    for image_dir in image_dirs:
        resize_images(join(source_path, image_dir), join(target_path, image_dir))


if __name__ == '__main__':
    # resize_image(r'C:\Users\JasSu\Documents\UHD\Sem7\capstone_project\active_shooter_prediction\presentation\images\detect.png'
    #              ,r'C:\Users\JasSu\Documents\UHD\Sem7\capstone_project\active_shooter_prediction\presentation\images')
    resize_image(r"C:\Users\JasSu\Documents\UHD\stargazer\il_1588xN.4191887579_czhc.webp", r"C:\Users\JasSu\Documents\UHD\stargazer")
