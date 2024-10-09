from ultralytics import YOLO
import cv2
import time
import numpy as np
import datetime
#import sys
from gpiozero import DigitalOutputDevice

from gpiozero import Servo
#TODO: change the gpio factory to reduce jitter
    #from gpiozero.pins.pigpio import PiGPIOFactory
    #factory = PiGPIOFactory()
 
myGPIO=24
myCorrection=0.5
maxPW=(2.0+myCorrection)/1000
minPW=(1.0-myCorrection)/1000
servo = Servo(myGPIO,min_pulse_width=minPW,max_pulse_width=maxPW)
servo.min()
#camera_index is the video device number of the camera 
camera_index = 0
relay = DigitalOutputDevice(18, active_high=True, initial_value=False) # Initialize GPIO 18
# Load a pretrained YOLO model 
model = YOLO("best.pt")
cam = cv2.VideoCapture(camera_index)
lastmean=0
grayImage1 =0 # This always processes the second image
while True:
    ret, image1 = cam.read()
    time.sleep(1)
    grayImage1= cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    grayImage2 = grayImage1
    motion = np.abs(np.mean(grayImage1) - lastmean)
    lastmean= np.mean(grayImage1)

    if motion>1:
        print(motion)
        now = datetime.datetime.now().isoformat()
        results = model(image1, stream=False)
        with open('summary.txt','a') as s:
            print(now +"\t",file=s, end="")
            #This generates a tab separated file for further analysis
            for conf in results:
                Outlist = list()
                count =  range(0,3)
                for n in count:
                    pick = results[0].probs.top5[0]
                    pickname = conf.names[pick]
                    thisidx = results[0].probs.top5[n]
                    name = conf.names[thisidx]
                    confvalue = results[0].probs.top5conf[n].item()
                    output = name + "\t" + str(f"{confvalue:.2%}")
                    Outlist.append(output)
                Outlist.sort()
                
                print(pickname +"\t",file=s, end ="")
                for thisitem in Outlist:
                         print(thisitem +"\t",file=s,end="")
            filename = "images/" + pickname + now + ".jpg"
            print(filename,file=s)
 
        with open('file2.txt', 'a') as f:

            for result in results:

                nameidX = result.probs.top1
                names = result.names[nameidX]
                confidence = result.probs.top1conf.cpu()
                print(now)
                cv2.imwrite(filename,image1)
                if nameidX == 2 and confidence > .75:

                    print("***ALARM***",file=f)                    
                    print(filename, file=f)

                    for x in range(20):
                        relay.on() # This will turn relay on
                        servo.max()
                        print(relay.is_active)
                        time.sleep(.2)
                        relay.off() # turn the relay off
                        servo.min()
                        print(relay.is_active)
                        time.sleep(.2)


    
