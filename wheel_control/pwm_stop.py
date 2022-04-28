import RPi.GPIO as GPIO
import time

ENA = 33  # use ENB = 32 for the other channel
IN1 = 37  # use IN3 = 35 for the other channel
IN2 = 38  # use IN3 = 36 for the other channel


def main():
    # Board pin-numbering scheme
    GPIO.setmode(GPIO.BOARD)
    # set pins as output with optional initial states
    GPIO.setup(ENA, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(IN1, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(IN2, GPIO.OUT, initial=GPIO.LOW)
    p = GPIO.PWM(ENA, 50)  # PWM values: 0-100
    # val = 50
    # incr = 5
    # p.start(val)
    print("stopping PWM")
    # for _ in range()
    while True:
        p.stop()
        GPIO.cleanup()
    


if __name__ == "__main__":
    main()
