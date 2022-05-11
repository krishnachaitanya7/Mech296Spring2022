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

if __name__ == "__main__":
    cap = cv2.VideoCapture(gsp(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        try:

            while True:
                ret, img = cap.read()

        finally:
            cap.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open CSI camera")
