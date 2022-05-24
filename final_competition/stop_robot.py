from wheel_control import MotionController
from time import sleep

if __name__ == "__main__":
    motion_controller = MotionController()
    for _ in range(0, 10):
        motion_controller.fire()
        sleep(0.1)
