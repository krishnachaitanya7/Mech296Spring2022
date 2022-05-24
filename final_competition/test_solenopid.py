from wheel_control import SolenoidController
from time import sleep

if __name__ == "__main__":
    solenoid_controller = SolenoidController()
    # for _ in range(0, 10):
    #     solenoid_controller.fire()
    #     sleep(5)
    solenoid_controller.machine_gun()
