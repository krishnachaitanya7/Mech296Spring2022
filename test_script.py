import sys

sys.path.append("/home/robotvision/opencv/build/lib/python3")
import cv2

image = cv2.imread("/home/robotvision/Downloads/lena.jpg")
# Window name in which image is displayed
window_name = "image"
# Using cv2.imshow() method
# Displaying the image
cv2.imshow(window_name, image)
# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)
