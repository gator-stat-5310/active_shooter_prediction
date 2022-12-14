# import required libraries
import cv2
import pathlib
import os







def detect_bounding_box(image_path):
    # read the input image

    img = cv2.imread(image_path)
    image_path_as_path = pathlib.Path(image_path)
    parent_dir_name = image_path_as_path.parent.stem

    # convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply thresholding on the gray image to create a binary image
    ret,thresh = cv2.threshold(gray, 127, 255, 0)

    # find the contours
    contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # compute the bounding rectangle of the contour
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        img = cv2.drawContours(img, [contour], 0, (0, 255, 255), 2)
        # draw the bounding rectangle
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # display the image with bounding rectangle drawn on it
    cv2.imshow("Bounding Rectangle", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image = r"C:\Users\JasSu\Documents\UHD\stargazer\il_1588xN.4191887579_czhc_640.jpg"
    detect_bounding_box(image)