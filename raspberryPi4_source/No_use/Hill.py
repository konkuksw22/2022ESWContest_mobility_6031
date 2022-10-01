from gpiozero import RGBLED
import time
import board
import busio
import adafruit_adxl34x

def adxlgetnum(param):
    acc = adafruit_adxl34x.ADXL345(param)
    acc.enable_freefall_detection(threshold=10, time = 25)
    acc.enable_motion_detection(threshold=18)
    acc.enable_tap_detection(tap_count = 1, threshold=20, duration=50, latency=20, window=255)

    return acc

def HillMode(freefall):
    # Object Detection Start
    # Output : List(String)
    obstacle = {"motorcycle", "bicycle", "car", "person"}
    objectname = detect()
    if objectname in obstacle:
        print("Stop")
    else :
        print("Nothing")
    


def detect():
    detectobject = {"motorcycle", "bicycle", "car", "person"}
    return detectobject

if __name__=="__main__":
    i2c = busio.I2C(board.SCL, board.SDA)
    accelerometer = adxlgetnum(i2c)    
    led = RGBLED(red=6, green=13, blue = 19)
    while True:
        x, y, z = accelerometer.acceleration
        print("%f %f %f"%(x, y, z))
        if y >= 5.0:
            led.color = (1, 0, 1)
        else:
            led.color = (0, 0, 0)
        print("Dropped: %s"%accelerometer.events["freefall"])
        print("Tapped: %s"%accelerometer.events["tap"])
        print("Motion detected: %s"%accelerometer.events["motion"])
        time.sleep(0.5)

    time.sleep(0.5)
