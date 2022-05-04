from wheel_control import MotionController
import time
import cv2
import signal
from time import sleep
import PySimpleGUI as sg

mc = MotionController()
# Detect the ctrl+c key combo
def ctrlc_handler(signal, frame):
    print("\nYou pressed Ctrl+C!")
    for _ in range(2):
        mc.stop()
    exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, ctrlc_handler)
    sg.theme("SandyBeach")
    # Very basic window.
    # Return values using
    # automatic-numbered keys
    layout = [
        [sg.Text("Please enter the parameters")],
        [sg.Text("Left PWM", size=(15, 1)), sg.InputText()],
        [sg.Text("Right PWM", size=(15, 1)), sg.InputText()],
    ]
    window = sg.Window("Simple data entry window", layout)
    while True:
        _, values = window.read(timeout=100)
        try:
            pwm_left = int(values[0])
        except:
            pwm_left = 0
        try:
            pwm_right = int(values[1])
        except:
            pwm_right = 0

        start_time = time.time()
        while time.time() - start_time < 2.0:
            print(f"Turning left")
            mc.go_left_and_right(pwm_left, pwm_right)
            sleep(2.0)

        # start_time = time.time()
        # while time.time() - start_time < 2:
        #     print(f"Stopping")
        #     mc.stop()
