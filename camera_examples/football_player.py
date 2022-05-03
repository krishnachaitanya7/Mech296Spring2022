import os
import cv2
import numpy as np
import glob
import jetson.utils
from gsp import gstreamer_pipeline as gsp


class FootBallPlayer:
    def __init__(self) -> None:
        self.goal_detected = False
        # self.camera_loop()

    def detect_checkerboard(self, img, draw_img=False):
        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (3, 3), None)
        # if corners are found, draw them
        if ret == True:
            self.goal_detected = True
            if draw_img:
                cv2.drawChessboardCorners(img, (1, 1), corners, ret)
                cv2.imshow("Goal Post", img)
                keyCode = cv2.waitKey(1) & 0xFF
                if keyCode == 27 or keyCode == ord("q"):
                    cv2.destroyAllWindows()
                    exit(0)
        else:
            print(f"No goal post detected")

    def camera_loop(self):
        window_title = "CSI Camera"
        # find the number of images in camera_calibrartion_images
        image_path = os.path.join(os.path.dirname(__file__), "camera_calibration_images")
        image_count = len(glob.glob1(image_path, "*.jpg"))
        image_count += 1
        # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
        cap = cv2.VideoCapture(gsp(flip_method=0), cv2.CAP_GSTREAMER)
        if cap.isOpened():
            try:
                while True:
                    ret, img = cap.read()
                    cv2.imshow(window_title, img)
                    # detect keypress on the image window
                    self.detect_checkerboard(img)

                    keyCode = cv2.waitKey(1) & 0xFF
                    # Stop the program on the ESC key or 'q'
                    if keyCode == 27 or keyCode == ord("q"):
                        break
                    elif keyCode == ord("c"):
                        # save the image
                        print("Saving image {}".format(image_count))
                        image_count += 1
                        cv2.imwrite(os.path.join(image_path, f"{image_count}.jpg"), img)
            finally:
                cap.release()
                cv2.destroyAllWindows()
        else:
            print("Error: Unable to open CSI camera")


if __name__ == "__main__":
    fb = FootBallPlayer()
    # fb.camera_loop()
    goal_image = cv2.imread("/home/robotvision/PycharmProjects/Mech296Spring2022/test_images/20220428_074200.jpg")
    fb.detect_checkerboard(goal_image, draw_img=True)
