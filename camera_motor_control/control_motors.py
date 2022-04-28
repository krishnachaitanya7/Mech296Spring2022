from wheel_control import MotionController
import time

if __name__ == "__main__":
    mc = MotionController()
    start_time = time.time()
    while time.time() - start_time < 10:
        mc.go_forward(50)
        time.sleep(0.1)
    while time.time() - start_time < 5:
        mc.stop()
    while time.time() - start_time < 10:
        mc.go_backward(50)
        time.sleep(0.1)
