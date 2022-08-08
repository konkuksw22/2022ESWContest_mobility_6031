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


if __name__=="__main__":
    i2c = busio.I2C(board.SCL, board.SDA)
    accelerometer = adxlgetnum(i2c)    

    while True:
        print("%f %f %f"%accelerometer.acceleration)
        print("Dropped: %s"%accelerometer.events["freefall"])
        print("Tapped: %s"%accelerometer.events["tap"])
        print("Motion detected: %s"%accelerometer.events["motion"])

    time.sleep(0.5)
