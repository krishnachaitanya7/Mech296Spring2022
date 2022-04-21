"""
Execute these commands on startup
busybox devmem 0x700031fc 32 0x45
busybox devmem 0x6000d504 32 0x2
busybox devmem 0x70003248 32 0x46
busybox devmem 0x6000d100 32 0x00
"""
import RPi.GPIO as GPIO
import time


LF = 35  
LB = 36  
RF = 37
RB = 38
left_wheel = 32
right_wheel = 33

def main():
    # Pin Setup:
    # Board pin-numbering scheme
    GPIO.setmode(GPIO.BOARD)
    # set pin as an output pin with optional initial state of HIGH
    GPIO.setup(left_wheel, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(right_wheel, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(LB, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(LF, GPIO.OUT, initial=GPIO.HIGH)
    GPIO.setup(RB, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(RF, GPIO.OUT, initial=GPIO.HIGH)
    left_wheel_pwm = GPIO.PWM(left_wheel, 100)
    right_wheel_pwm = GPIO.PWM(right_wheel, 100)
    val = 10
    incr = 10
    left_wheel_pwm.start(val)
    right_wheel_pwm.start(val)

    print("PWM running. Press CTRL+C to exit.")
    try:
        while True:
            time.sleep(1.0)
            if val >= 100:
                incr = -incr
            if val <= 0:
                incr = -incr
            val += incr
            left_wheel_pwm.ChangeDutyCycle(val)
            right_wheel_pwm.ChangeDutyCycle(val)
            print(f"Duty Cycle: {val}")
    finally:
        left_wheel_pwm.stop()
        right_wheel_pwm.stop()
        GPIO.cleanup()

if __name__ == '__main__':
    main()