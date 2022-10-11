import RPi.GPIO as GPIO
import time
import board
import busio
import adafruit_adxl34x
import neopixel
from socket import *
import sys

host = '203.###.###.##'
port = int(sys.argv[1])

pixel_pin = board.D10
num_pixels = 10
ORDER = neopixel.GRB

def adxlgetnum(param):
    acc = adafruit_adxl34x.ADXL345(param)
    acc.enable_freefall_detection(threshold=10, time = 25)
    acc.enable_motion_detection(threshold=18)
    acc.enable_tap_detection(tap_count = 1, threshold=20, duration=50, latency=20, window=255)

    return acc

def rightMode(src):
    if src[1]=='1':
        print("Danger! : People is still crossing")
        pixels[8]=(150,80,0)  
    time.sleep(1)    
    pixels.fill((0,0,0))    
            
    print("-----------------------")
    
def detect(src,pixels,num):
    detectobject = ["car", "person", "bicycle", "motorcycle"]
    for l in range(len(detectobject)-1):
        if src[l] == '1' :
            if num==0:
                pixels[l]=(200,255,0)
                
                print("A : Danger! : " + detectobject[l] + " is detected")
            elif num ==1:
                pixels[l+4]=(0,120,255)
                
                print("H : Danger! : " + detectobject[l] + " is detected")
    time.sleep(1)
    pixels.fill((0,0,0))        
    print("-----------------------")


if __name__=="__main__":
    i2c = busio.I2C(board.SCL, board.SDA)
    accelerometer = adxlgetnum(i2c)    
    #led = RGBLED(red=6, green=13,blue=19)
    pixels = neopixel.NeoPixel( pixel_pin, num_pixels, brightness=0.2)

    serverSocket = socket(AF_INET, SOCK_STREAM)
    serverSocket.bind((host,port))
    serverSocket.listen()
    connectionSocket,addr=serverSocket.accept()

    GPIO.setwarnings(False) 
    GPIO.setmode(GPIO.BCM) 
    GPIO.setup(18, GPIO.IN)

    token=[0,0,0,0,0,0,0,0,0,0]

    while True:
        token=connectionSocket.recv(1024)
        token=token.decode('utf-8')

        pillar=token[:3]
        hill=token[3:7]
        right=token[7:]
        
        # Apillar Detect
        detect(pillar,pixels,0)
        #pixels.fill((0,0,0))

        # Hill Detect
        x, y, z = accelerometer.acceleration
        if y >= 4.9 :
            detect(hill,pixels,1)     
            print("Hill")      
        else:
            print('Not Hill')

        # Right Detect
        if GPIO.input(18) == 1 and right[0] =='1':
            pixels[7]=(255,0,0)
            print('Right')
            rightMode(right)
                 
        if 0xFF == ord('q'):
            pixels.fill((0,0,0))
            break
        time.sleep(1)
        pixels.fill((0, 0, 0))
        pixels.show()

    print('End')
    serverSocket.close()
    time.sleep(0.5)