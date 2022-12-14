# import required libraries
import cv2
import pathlib
import os


def build_bounding_box(index, image_path, outputdir, write_all_contours = False):
    # read the input image

    img = cv2.imread(image_path)
    image_path_as_path = pathlib.Path(image_path)
    parent_dir_name = image_path_as_path.parent.stem
    image_label_path = os.path.join(outputdir, parent_dir_name, f"{image_path_as_path.stem}.txt")
    os.makedirs(pathlib.Path(image_label_path).parent, exist_ok=True)

    # convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply thresholding on the gray image to create a binary image
    ret,thresh = cv2.threshold(gray, 127, 255, 0)

    # find the contours
    contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # compute the bounding rectangle of the contour
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if w>x and h >y:
            print(x, y, w, h)
            with open(image_label_path, "wt") as file:
                file.write(f"{index} {((w - x) / 2)/w} {((h - y) / 2)/h} {w/w} {h/h}")
                break
        else:
            print(f"ignoring {x}, {y}, {w}, {h}")
        if x >h or y >h:
            img = cv2.drawContours(img, [contour], 0, (0, 255, 255), 2)

            # draw the bounding rectangle
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # display the image with bounding rectangle drawn on it
            cv2.imshow("Bounding Rectangle", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    root_path = r'C:\Users\JasSu\Documents\UHD\Sem7\capstone_project\active_shooter_prediction\yolov5\data_images'
    root_output_path = r'C:\Users\JasSu\Documents\UHD\Sem7\capstone_project\active_shooter_prediction\yolov5\data_labels'
    os.makedirs(root_output_path, exist_ok=True)
    image_dirs = {"assault_rifles":0, "machine_guns":1, "pistols":2, "shot_guns":3}
    for image_dir, index in image_dirs.items():
        for root_, _, file_names in os.walk(os.path.join(root_path, image_dir)):
            for file_name in file_names:
                print(os.path.join(root_, file_name))
                build_bounding_box(index, os.path.join(root_, file_name), root_output_path)