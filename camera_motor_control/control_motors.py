from wheel_control import MotionController
import time
import cv2

mc = MotionController()

if __name__ == "__main__":
    start_time = time.time()
    while time.time() - start_time < 10:
        print(f"Going forwards")
        mc.go_forward(90)
        time.sleep(0.1)
    start_time = time.time()
    while time.time() - start_time < 5:
        print(f"Stopping")
        mc.stop()
    start_time = time.time()
    while time.time() - start_time < 10:
        print(f"Going backwards")
        mc.go_backward(90)
        time.sleep(0.1)
