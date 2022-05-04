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
        self.RF = 35
        self.RB = 36
        self.LF = 37
        self.LB = 38
        self.left_wheel = 33
        self.right_wheel = 32
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
        self._left_duty_cycle = 10
        self._right_duty_cycle = 10

    @property
    def left_duty_cycle(self):
        return self._left_duty_cycle

    @left_duty_cycle.setter
    def duty_cycle(self, value):
        self._left_duty_cycle = value

    @property
    def right_duty_cycle(self):
        return self._right_duty_cycle

    @right_duty_cycle.setter
    def right_duty_cycle(self, value):
        self._right_duty_cycle = value

    def set_wheel_alignment(self, LF_state, LB_state, RF_state, RB_state):
        self.LF_state = LF_state
        self.LB_state = LB_state
        self.RF_state = RF_state
        self.RB_state = RB_state
        GPIO.output(self.LF, self.LF_state)
        GPIO.output(self.LB, self.LB_state)
        GPIO.output(self.RF, self.RF_state)
        GPIO.output(self.RB, self.RB_state)

    def go_left_and_right(self, left_duty_cycle, right_duty_cycle):
        # Whenever a wheel needs to go back: GPIO.LOW, GPIO.HIGH
        # Whenever a wheel needs to go forward: GPIO.HIGH, GPIO.LOW
        if left_duty_cycle < 0 and right_duty_cycle < 0:
            self.set_wheel_alignment(GPIO.LOW, GPIO.HIGH, GPIO.LOW, GPIO.HIGH)
        elif left_duty_cycle < 0 and right_duty_cycle > 0:
            self.set_wheel_alignment(GPIO.LOW, GPIO.HIGH, GPIO.HIGH, GPIO.LOW)
        elif left_duty_cycle > 0 and right_duty_cycle < 0:
            self.set_wheel_alignment(GPIO.HIGH, GPIO.LOW, GPIO.LOW, GPIO.HIGH)
        else:
            # Both are greater than 0
            self.set_wheel_alignment(GPIO.HIGH, GPIO.LOW, GPIO.HIGH, GPIO.LOW)

        self._left_duty_cycle = abs(left_duty_cycle)
        self._right_duty_cycle = abs(right_duty_cycle)
        self.left_wheel_pwm.start(self._left_duty_cycle)
        self.right_wheel_pwm.start(self._right_duty_cycle)

    def stop(self):
        self.left_wheel_pwm.stop()
        self.right_wheel_pwm.stop()
        self.set_wheel_alignment(GPIO.LOW, GPIO.LOW, GPIO.LOW, GPIO.LOW)
