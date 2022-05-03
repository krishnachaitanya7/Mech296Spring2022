from wheel_control import MotionController
import time
import cv2

mc = MotionController()

if __name__ == "__main__":
    for _ in range(4):
        start_time = time.time()
        while time.time() - start_time < 5:
            #d\print(f"Turning left")
            mc.go_left(80, 80)

        start_time = time.time()
        while time.time() - start_time < 2:
            #print(f"Stopping")
            mc.stop()
    # for _ in range(4):
    #     start_time = time.time()
    #     while time.time() - start_time < 0.41:
    #         print(f"Turning Right")
    #         mc.go_left(-100, 100)
    #     start_time = time.time()
    #     while time.time() - start_time < 2:
    #         print(f"Stopping")
    #         mc.stop()
    # for _ in range(4):
    #     start_time = time.time()
    #     while time.time() - start_time < 0.8:
    #         print(f"Turning Right")
    #         mc.go_left(100, 0)
    #     start_time = time.time()
    #     while time.time() - start_time < 2:
    #         print(f"Stopping")
    #         mc.stop()

    # while time.time() - start_time < 10:
    #     print(f"Going forwards")
    #     mc.go_forward(90)
    #     time.sleep(0.1)
    # start_time = time.time()
    # while time.time() - start_time < 5:
    #     print(f"Stopping")
    #     mc.stop()
    # start_time = time.time()
    # while time.time() - start_time < 10:
    #     print(f"Going backwards")
    #     mc.go_backward(90)
    #     time.sleep(0.1)
