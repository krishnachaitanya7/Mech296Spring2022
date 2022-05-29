from wheel_control import MotionController
from time import sleep
from wheel_control import SolenoidController

if __name__ == "__main__":
    solenoid_controller = SolenoidController()
    motion_controller = MotionController()
    for _ in range(0, 10):
        motion_controller.stop()
        solenoid_controller.stop()
        sleep(0.1)

    # solenoid_controller.machine_gun()
