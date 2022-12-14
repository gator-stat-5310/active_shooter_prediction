import pathlib
import os


def split_training_test(src_images_root_path, src_labels_root_path, target_image_training_path, target_image_test_path,
                            target_label_training_path, target_label_test_path):

    images_=[]
    labels_=[]
    for image_root, _, images in os.walk(src_images_root_path):
        for image in images:
            images_.append(os.path.join(image_root, image))

    for label_root, _, labels in os.walk(src_labels_root_path):
        for label in labels:
            labels_.append(os.path.join(label_root, label))

    for index, images_labels_path in enumerate(zip(images_, labels_)):
        image_path, label_path = images_labels_path
        image_file_path_as_path = pathlib.Path(image_path)
        label_file_path_as_path = pathlib.Path(label_path)
        if index%5==0:
            image_file_path_as_path.rename(os.path.join(target_image_test_path, image_file_path_as_path.name))
            label_file_path_as_path.rename(os.path.join(target_label_test_path, label_file_path_as_path.name))
        else:
            image_file_path_as_path.rename(os.path.join(target_image_training_path, image_file_path_as_path.name))
            label_file_path_as_path.rename(os.path.join(target_label_training_path, label_file_path_as_path.name))


if __name__ == '__main__':
    src_root_path = r'C:\Users\JasSu\Documents\UHD\Sem7\capstone_project\active_shooter_prediction\yolov5'
    src_images_root_path = os.path.join(src_root_path, 'data_images')
    src_label_root_path = os.path.join(src_root_path, 'data_labels')
    target_root_path = r'C:\Users\JasSu\Documents\UHD\Sem7\capstone_project\active_shooter_prediction\yolov5\data'
    target_image_path = os.path.join(target_root_path, r'images')
    target_image_training_path = os.path.join(target_image_path, 'train')
    target_image_test_path = os.path.join(target_image_path, 'val')
    target_label_path = os.path.join(target_root_path, r'label')
    target_label_training_path = os.path.join(target_label_path, 'train')
    target_label_test_path = os.path.join(target_label_path, 'val')

    os.makedirs(target_label_training_path, exist_ok=True)
    os.makedirs(target_label_test_path , exist_ok=True)

    os.makedirs(target_image_training_path, exist_ok=True)
    os.makedirs(target_image_test_path , exist_ok=True)

    image_dirs = ["assault_rifles", "machine_guns", "pistols", "shot_guns"]
    for image_dir in image_dirs:
        split_training_test(src_images_root_path, src_label_root_path, target_image_training_path,target_image_test_path,
                            target_label_training_path, target_label_test_path)