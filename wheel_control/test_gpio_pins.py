from wheel_control import MotionController
from time import sleep

if __name__ == "__main__":
    mc = MotionController()
    while True:
        mc.go_left_and_right(25, 35)
        sleep(0.5)
        # mc.stop()
