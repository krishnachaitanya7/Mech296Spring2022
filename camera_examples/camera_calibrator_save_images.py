import cv2
import jetson.inference
import jetson.utils
from gsp import gstreamer_pipeline as gsp
import os
import glob


def main():
    window_title = "CSI Camera"
    # find the number of images in camera_calibrartion_images
    image_path = os.path.join(os.path.dirname(__file__), "dataset_images")
    image_count = len(glob.glob1(image_path, "*.jpg"))
    image_count = 1
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    cap = cv2.VideoCapture(gsp(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        try:
            while True:
                ret, img = cap.read()
                cv2.imshow(window_title, img)
                # detect keypress on the image window

                keyCode = cv2.waitKey(1) & 0xFF
                # Stop the program on the ESC key or 'q'
                if keyCode == 27 or keyCode == ord("q"):
                    break
                elif keyCode == ord("c"):
                    # save the image
                    print("Saving image {}".format(image_count))
                    cv2.imwrite(os.path.join(image_path, f"{image_count}.jpg"), img)
                    image_count += 1
        finally:
            cap.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open CSI camera")


if __name__ == "__main__":
    main()
