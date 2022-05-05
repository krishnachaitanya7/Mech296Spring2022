import os
import cv2
import numpy as np
import glob
import jetson.utils
from gsp import gstreamer_pipeline as gsp
from math import pi
import PySimpleGUI as sg
from wheel_control import MotionController
import time


class GoToCenter:
    def __init__(self, initilize_gui=False, show_plots=False):
        self.initilize_gui = initilize_gui
        if self.initilize_gui:
            sg.theme("SandyBeach")
            # Very basic window.
            # Return values using
            # automatic-numbered keys
            layout = [
                [sg.Text("Please enter the parameters")],
                [sg.Text("Lower H", size=(15, 1)), sg.InputText()],
                [sg.Text("Lower S", size=(15, 1)), sg.InputText()],
                [sg.Text("Lower V", size=(15, 1)), sg.InputText()],
                [sg.Text("Higher H", size=(15, 1)), sg.InputText()],
                [sg.Text("Higher S", size=(15, 1)), sg.InputText()],
                [sg.Text("Higher V", size=(15, 1)), sg.InputText()],
            ]
            self.window = sg.Window("Simple data entry window", layout)
        self.show_plots = show_plots
        if show_plots:
            cv2.namedWindow("Green Mask", cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow("Blue Mask", cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        self.canny_low = None
        self.canny_high = None
        self.min_line_length = None
        self.max_line_gap = None
        self.low_thresh = None
        self.high_thresh = None
        self.bottom_color = None
        self.mc = MotionController()
        self.go_to_center()

    def read_gui_input(self):
        # Read the values from the GUI
        _, values = self.window.read(timeout=100)
        try:
            lower_h = int(values[0])
        except:
            lower_h = 94
        try:
            lower_s = int(values[1])
        except:
            lower_s = 80
        try:
            lower_v = int(values[2])
        except:
            lower_v = 2
        try:
            higher_h = int(values[3])
        except:
            higher_h = 126
        try:
            higher_s = int(values[4])
        except:
            higher_s = 255
        try:
            higher_v = int(values[5])
        except:
            higher_v = 255
        return lower_h, lower_s, lower_v, higher_h, higher_s, higher_v

    def get_blue_green_masks(self, img):
        cnt = None
        countour_area = None
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if self.initilize_gui:
            lower_h, lower_s, lower_v, higher_h, higher_s, higher_v = self.read_gui_input()
        else:
            lower_h, lower_s, lower_v, higher_h, higher_s, higher_v = 17, 150, 90, 126, 255, 255
        # define range of blue color in HSV
        lower_blue = np.array([lower_h, lower_s, lower_v])
        upper_blue = np.array([higher_h, higher_s, higher_v])
        # Threshold the HSV image to get only blue colors
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # define range of green color in HSV
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([70, 255, 255])
        # Threshold the HSV image to get only green colors
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        green_percent, blue_percent = self.get_blue_green_percentage(blue_mask, mask_green)
        if self.get_bottom_color(blue_mask, mask_green) == "green":
            self.bottom_color = "green"
            # draw countours on green mask
            contours, hierarchy = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # find the biggest contour
            if len(contours) > 0:
                cnt = max(contours, key=cv2.contourArea)
        else:
            self.bottom_color = "blue"
            contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # find the biggest contour
            if len(contours) > 0:
                cnt = max(contours, key=cv2.contourArea)
        if cnt is not None:
            countour_area = cv2.contourArea(cnt)
            print(f"Countour Area is {countour_area}")
        if self.show_plots:
            if cnt is not None:
                img_copy = img.copy()
                cv2.drawContours(img_copy, [cnt], -1, (0, 255, 0), 3)
                cv2.imshow("Mask with contours", img_copy)
                keyCode = cv2.waitKey(1) & 0xFF
                # Stop the program on the ESC key or 'q'
                if keyCode == 27 or keyCode == ord("q"):
                    raise Exception("User exit")

        print(f"Green: {green_percent}%, Blue: {blue_percent}%")
        if self.show_plots:
            cv2.imshow("Blue Mask", blue_mask)
            cv2.imshow("Green Mask", mask_green)
            keyCode = cv2.waitKey(1) & 0xFF
            # Stop the program on the ESC key or 'q'
            if keyCode == 27 or keyCode == ord("q"):
                raise Exception("User exit")
        if countour_area is not None:
            return (blue_mask, mask_green, self.bottom_color, countour_area)
        else:
            return (blue_mask, mask_green, self.bottom_color, None)

    def get_bottom_color(self, blue_mask, mask_green):
        # Bitwise-AND mask and original image
        # Get bottom row of green mask
        bottom_row_green = mask_green[-1, :]
        # Get bottom row of blue mask
        bottom_row_blue = blue_mask[-1, :]
        # count the number of green pixels in the bottom row
        num_green_pixels = np.sum(bottom_row_green) / 255
        # count the number of blue pixels in the bottom row
        num_blue_pixels = np.sum(bottom_row_blue) / 255
        if num_green_pixels > num_blue_pixels:
            return "green"
        else:
            return "blue"

    def get_blue_green_percentage(self, blue_mask, green_mask):
        # Calculate the percentage of green pixels in the entire mask
        num_green_pixels = (np.sum(green_mask) / 255) / (green_mask.shape[0] * green_mask.shape[1]) * 100
        num_blue_pixels = (np.sum(blue_mask) / 255) / (blue_mask.shape[0] * blue_mask.shape[1]) * 100
        return num_green_pixels, num_blue_pixels

    def turn_left(self):
        self.mc.go_left_and_right(-60, 60)
        time.sleep(0.05)
        self.mc.stop()
        time.sleep(0.5)

    def move_forward(self):
        self.mc.go_left_and_right(60, 60)
        time.sleep(0.5)
        self.mc.stop()
        time.sleep(0.5)

    def go_to_center(self):
        self.canny_low, self.canny_high, self.min_line_length, self.max_line_gap, self.low_thresh, self.high_thresh = (
            190,
            200,
            100,
            5,
            150,
            255,
        )
        cap = cv2.VideoCapture(gsp(flip_method=0), cv2.CAP_GSTREAMER)
        if cap.isOpened():
            try:
                while True:
                    ret, img = cap.read()
                    blue_mask, mask_green, self.bottom_color, countour_area = self.get_blue_green_masks(img)
                    if countour_area is not None:
                        if self.bottom_color == "blue":
                            print(f"Bottom color is blue")
                            if countour_area > 23000:
                                # while self.get_bottom_color(blue_mask, mask_green) == "blue":
                                self.move_forward()
                            else:
                                self.turn_left()
                        else:
                            print(f"Bottom color is green")
                            self.mc.stop()

                    if self.show_plots:
                        cv2.imshow("CSI Camera", img)
                        keyCode = cv2.waitKey(1) & 0xFF
                        # Stop the program on the ESC key or 'q'
                        if keyCode == 27 or keyCode == ord("q"):
                            raise Exception("User exit")
            finally:
                cap.release()
                cv2.destroyAllWindows()
        else:
            print("Error: Unable to open CSI camera")


if __name__ == "__main__":
    GoToCenter(initilize_gui=False, show_plots=True)
