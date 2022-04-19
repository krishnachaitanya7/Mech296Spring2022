import RPi.GPIO as GPIO
import time

ENA = 32  # use ENB = 32 for the other channel
ENB = 33
IN1 = 35  # use IN3 = 37 for the other channel
IN2 = 36  # use IN4 = 38 for the other channel
IN3 = 37
IN4 = 38


def main():
    # Board pin-numbering scheme
    GPIO.setmode(GPIO.BOARD)
    # set pins as output with optional initial states
    # GPIO.setup(ENA, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(ENA, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(ENB, GPIO.OUT, initial=GPIO.HIGH)
    GPIO.setup(IN1, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(IN2, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(IN3, GPIO.OUT, initial=GPIO.HIGH)
    GPIO.setup(IN4, GPIO.OUT, initial=GPIO.LOW)
    p = GPIO.PWM(ENB, 100)  # PWM values: 0-100
    val = 30
    incr = 10
    p.start(val)
    print("PWM running, abort with CTRL+C")
    try:
        while True:
            time.sleep(5.0)
            if val >= 100 or val <= 0:
                incr = -incr
            val += incr
            
            print(f"Duty Cycle: {val}")
            p.ChangeDutyCycle(val)
    finally:
        print("stopping PWM")
        p.stop()
        GPIO.setup(ENB, GPIO.OUT, initial=GPIO.LOW)
        GPIO.cleanup()


if __name__ == "__main__":
    main()
