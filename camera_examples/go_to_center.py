import os
import cv2
import numpy as np
import glob
import jetson.utils
from gsp import gstreamer_pipeline as gsp
from math import pi
import PySimpleGUI as sg


class GoToCenter:
    def __init__(self, initilize_gui=False, show_plots=False) -> None:
        self.initilize_gui = initilize_gui
        if self.initilize_gui:
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
        self.go_to_center()

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

    def get_blue_green_masks(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # define range of blue color in HSV
        lower_blue = np.array([94, 80, 2])
        upper_blue = np.array([126, 255, 255])
        # Threshold the HSV image to get only blue colors
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # define range of green color in HSV
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([70, 255, 255])
        # Threshold the HSV image to get only green colors
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        green_percent, blue_percent = self.get_blue_green_percentage(blue_mask, mask_green)
        print(f"Green: {green_percent}%, Blue: {blue_percent}%")
        if self.show_plots:
            cv2.imshow("Blue Mask", blue_mask)
            cv2.imshow("Green Mask", mask_green)
            keyCode = cv2.waitKey(1) & 0xFF
            # Stop the program on the ESC key or 'q'
            if keyCode == 27 or keyCode == ord("q"):
                raise Exception("User exit")
        return blue_mask, mask_green

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

    def go_to_center(self):
        if self.initilize_gui:
            (
                self.canny_low,
                self.canny_high,
                self.min_line_length,
                self.max_line_gap,
                self.low_thresh,
                self.high_thresh,
            ) = self.read_gui_input()
        else:
            (
                self.canny_low,
                self.canny_high,
                self.min_line_length,
                self.max_line_gap,
                self.low_thresh,
                self.high_thresh,
            ) = (190, 200, 100, 5, 150, 255)
        cap = cv2.VideoCapture(gsp(flip_method=0), cv2.CAP_GSTREAMER)
        if cap.isOpened():
            try:
                while True:
                    ret, img = cap.read()
                    masks = self.get_blue_green_masks(img)
                    bottom_color = self.get_bottom_color(*masks)
                    print(f"Bottom color: {bottom_color}")
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
