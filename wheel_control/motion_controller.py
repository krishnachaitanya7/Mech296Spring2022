"""
Execute these commands on startup
busybox devmem 0x700031fc 32 0x45
busybox devmem 0x6000d504 32 0x2
busybox devmem 0x70003248 32 0x46
busybox devmem 0x6000d100 32 0x00
"""
import RPi.GPIO as GPIO


class MotionController:
    def __init__(self):
        # Pin Setup:
        GPIO.setmode(GPIO.BOARD)
        self.LF = 35
        self.LB = 36
        self.RF = 37
        self.RB = 38
        self.left_wheel = 32
        self.right_wheel = 33
        self.LB_state = GPIO.LOW
        self.LF_state = GPIO.HIGH
        self.RB_state = GPIO.LOW
        self.RF_state = GPIO.HIGH
        GPIO.setup(self.left_wheel, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.right_wheel, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.LB, GPIO.OUT, initial=self.LB_state)
        GPIO.setup(self.LF, GPIO.OUT, initial=self.LF_state)
        GPIO.setup(self.RB, GPIO.OUT, initial=self.RB_state)
        GPIO.setup(self.RF, GPIO.OUT, initial=self.RF_state)
        self.left_wheel_pwm = GPIO.PWM(self.left_wheel, 100)
        self.right_wheel_pwm = GPIO.PWM(self.right_wheel, 100)
        self.duty_cycle = 10

    @property
    def duty_cycle(self):
        return self._duty_cycle

    @duty_cycle.setter
    def duty_cycle(self, value):
        self.duty_cycle = value

    def set_wheel_alignment(self, LF_state, LB_state, RF_state, RB_state):
        self.LF_state = LF_state
        self.LB_state = LB_state
        self.RF_state = RF_state
        self.RB_state = RB_state
        GPIO.output(self.LF, self.LF_state)
        GPIO.output(self.LB, self.LB_state)
        GPIO.output(self.RF, self.RF_state)
        GPIO.output(self.RB, self.RB_state)

    def go_forward(self, duty_cycle=None):
        if duty_cycle is not None:
            self.duty_cycle = duty_cycle
        self.set_wheel_alignment(GPIO.HIGH, GPIO.LOW, GPIO.HIGH, GPIO.LOW)
        self.left_wheel_pwm.start(self.duty_cycle)
        self.right_wheel_pwm.start(self.duty_cycle)

    def go_backward(self, duty_cycle=None):
        if duty_cycle is not None:
            self.duty_cycle = duty_cycle
        self.set_wheel_alignment(GPIO.LOW, GPIO.HIGH, GPIO.LOW, GPIO.HIGH)
        self.left_wheel_pwm.start(self.duty_cycle)
        self.right_wheel_pwm.start(self.duty_cycle)

    def stop(self):
        self.left_wheel_pwm.stop()
        self.right_wheel_pwm.stop()
        self.set_wheel_alignment(GPIO.LOW, GPIO.LOW, GPIO.LOW, GPIO.LOW)


# def main():
#     # Pin Setup:
#     # Board pin-numbering scheme
#     GPIO.setmode(GPIO.BOARD)
#     # set pin as an output pin with optional initial state of HIGH
#     GPIO.setup(left_wheel, GPIO.OUT, initial=GPIO.LOW)
#     GPIO.setup(right_wheel, GPIO.OUT, initial=GPIO.LOW)
#     GPIO.setup(LB, GPIO.OUT, initial=GPIO.LOW)
#     GPIO.setup(LF, GPIO.OUT, initial=GPIO.HIGH)
#     GPIO.setup(RB, GPIO.OUT, initial=GPIO.LOW)
#     GPIO.setup(RF, GPIO.OUT, initial=GPIO.HIGH)
#     left_wheel_pwm = GPIO.PWM(left_wheel, 100)
#     right_wheel_pwm = GPIO.PWM(right_wheel, 100)
#     val = 10
#     incr = 10
#     left_wheel_pwm.start(val)
#     right_wheel_pwm.start(val)

#     print("PWM running. Press CTRL+C to exit.")
#     try:
#         while True:
#             time.sleep(1.0)
#             if val >= 100:
#                 incr = -incr
#             if val <= 0:
#                 incr = -incr
#             val += incr
#             left_wheel_pwm.ChangeDutyCycle(val)
#             right_wheel_pwm.ChangeDutyCycle(val)
#             print(f"Duty Cycle: {val}")
#     finally:
#         left_wheel_pwm.stop()
#         right_wheel_pwm.stop()
#         GPIO.cleanup()


# if __name__ == "__main__":
#     main()
