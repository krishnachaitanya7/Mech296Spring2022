import os
import cv2
import numpy as np
import glob
import jetson.utils
from gsp import gstreamer_pipeline as gsp
from math import pi
import PySimpleGUI as sg


class FootBallPlayer:
    def __init__(self) -> None:
        self.goal_detected = False
        # Setup the GUI
        sg.theme("SandyBeach")
        # Very basic window.
        # Return values using
        # automatic-numbered keys
        layout = [
            [sg.Text("Please enter the parameters")],
            [sg.Text("Canny Low", size=(15, 1)), sg.InputText()],
            [sg.Text("Canny High", size=(15, 1)), sg.InputText()],
            [sg.Text("Min Line Length", size=(15, 1)), sg.InputText()],
            [sg.Text("Max Line Gap", size=(15, 1)), sg.InputText()],
            [sg.Text("Min Threshold", size=(15, 1)), sg.InputText()],
            [sg.Text("Max Threshold", size=(15, 1)), sg.InputText()],
        ]
        self.window = sg.Window("Simple data entry window", layout)
        self.camera_loop1()

    def read_gui_input(self):
        # Read the values from the GUI
        _, values = self.window.read(timeout=100)
        try:
            canny_low = int(values[0])
        except:
            canny_low = 190
        try:
            canny_high = int(values[1])
        except:
            canny_high = 200
        try:
            min_line_length = int(values[2])
        except:
            min_line_length = 100
        try:
            max_line_gap = int(values[3])
        except:
            max_line_gap = 5
        try:
            low_thresh = int(values[4])
        except:
            low_thresh = 150
        try:
            high_thresh = int(values[5])
        except:
            high_thresh = 255
        return canny_low, canny_high, min_line_length, max_line_gap, low_thresh, high_thresh

    def camera_loop1(self):
        window_title = "CSI Camera"
        cap = cv2.VideoCapture(gsp(flip_method=0), cv2.CAP_GSTREAMER)
        if cap.isOpened():
            try:
                while True:
                    ret, img = cap.read()
                    # cv2.imshow(window_title, img)
                    # # detect keypress on the image window
                    # keyCode = cv2.waitKey(1) & 0xFF
                    # if keyCode == 27 or keyCode == ord("q"):
                    #     break
                    self.detect_bottom_color(img)
                    self.detect_checkerboard(img, draw_img=True)
            finally:
                cap.release()
                cv2.destroyAllWindows()
        else:
            print("Error: Unable to open CSI camera")

    def detect_checkerboard(self, img, draw_img=False):
        (
            canny_low,
            canny_high,
            min_line_length,
            max_line_gap,
            low_thresh,
            high_thresh,
        ) = self.read_gui_input()  # 190, 200, 100, 4
        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # find the chessboard corners
        img_grayscale_gb = cv2.GaussianBlur(gray, (5, 5), 0)
        img_grayscale_gb_canny = cv2.Canny(img_grayscale_gb, canny_low, canny_high)
        # find hough lines in the image
        lines = cv2.HoughLinesP(img_grayscale_gb_canny, 1, pi / 180, 50, None, min_line_length, max_line_gap)
        # draw the lines on the image
        # if draw_img and lines is not None:
        #     for line in lines:
        #         for x1, y1, x2, y2 in line:
        #             cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2 find countours
        _, binary = cv2.threshold(img_grayscale_gb, low_thresh, high_thresh, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # draw countours if detected
        if draw_img and contours is not None:
            for contour in contours:
                cv2.drawContours(img, contour, -1, (0, 255, 0), 2)

        cv2.imshow("Hough Lines", img)
        keyCode = cv2.waitKey(1) & 0xFF
        # Stop the program on the ESC key or 'q'
        if keyCode == 27 or keyCode == ord("q"):
            raise Exception("User stopped program")

    def detect_bottom_color(self, img):
        # RGB to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # define range of blue color in HSV
        lower_blue = np.array([94, 80, 2])
        upper_blue = np.array([126, 255, 255])
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # plot the mask image
        cv2.imshow("Blue Mask", mask)
        keyCode = cv2.waitKey(1) & 0xFF
        # Stop the program on the ESC key or 'q'
        if keyCode == 27 or keyCode == ord("q"):
            raise Exception("User stopped program")

        # define range of green color in HSV
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([70, 255, 255])
        # Threshold the HSV image to get only green colors
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        # plot the mask image
        cv2.imshow("Green Mask", mask_green)
        keyCode = cv2.waitKey(1) & 0xFF
        # Stop the program on the ESC key or 'q'
        if keyCode == 27 or keyCode == ord("q"):
            raise Exception("User stopped program")
        # print("sup")

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
    fb.camera_loop1()
