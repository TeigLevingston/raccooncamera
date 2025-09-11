from ultralytics import YOLO
import cv2
import time
import numpy as np
import datetime
import random
#import sys
from gpiozero import DigitalOutputDevice

from gpiozero import Servo
#TODO: change the gpio factory to reduce jitter
    #from gpiozero.pins.pigpio import PiGPIOFactory
    #factory = PiGPIOFactory()
 
myGPIO=24
myCorrection=0.5
#maxPW=(2.0+myCorrection)/1000
#minPW=(1.0-myCorrection)/1000
minPW=.0005
maxPW=.005
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
newstart=1
while True:
    ret, image1 = cam.read()
    if newstart ==1:
        newfilename = "images/" + datetime.datetime.now().isoformat() + ".jpg"
        print(newfilename)
        cv2.imwrite(newfilename,image1)
        for x in range(3):
            relay.on() # This will turn relay on
            print(relay.is_active)
            servo.max()
            print("max")
            time.sleep(.5)
            relay.off() # This will not work :( but
            print(relay.is_active)
            servo.min()
            print("min")
            servo.max()


            #servo.min()
        newstart=0
    time.sleep(1)

    grayImage1= cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    grayImage2 = grayImage1
    motion = np.abs(np.mean(grayImage1) - lastmean)
    lastmean= np.mean(grayImage1)
    ret, image0 = cam.read()
    #if motion<1:

    if motion>3:
        print("Motion:")
        print(motion)
        now = datetime.datetime.now().isoformat()
        print("Start: "+datetime.datetime.now().isoformat())
        results = model(image1, stream=False)
        print("end: " + datetime.datetime.now().isoformat())
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
            filename = "images/likelycat"  + now + ".jpg"
            filename0= "images/likelycat" + now + "_0.jpg"
            filename1= "images/likelycat" + now + "_1.jpg"
            print(filename,file=s)



            for result in results:

                nameidX = result.probs.top1
                coonconf = results[0].probs.top5conf[2].item()
                catconf = results[0].probs.top5conf[1].item()
                names = result.names[nameidX]
                confidence = result.probs.top1conf.cpu()
                print(now)
                print("Coon Confidence")
                print(coonconf)
                print(catconf)
                #
                if catconf> .15:
                    cv2.imwrite(filename,image1)
                    cv2.imwrite(filename0,image0)
                    ret, image2 = cam.read()
                    cv2.imwrite(filename1,image2)
                    print("High Cat Confidence")
                    print(coonconf)


                if nameidX == 2:
                    if confidence > .75:
                        filename = "images/raccoon"  + now + ".jpg"
                        filename0= "images/raccoon" + now + "_0.jpg"
                        filename1= "images/raccoon" + now + "_1.jpg"  
                        with open('file2.txt', 'a') as f:
                            print(now)
                            print("***ALARM***",file=f)                    
                            print(filename, file=f)
                            cv2.imwrite(filename,image1)
                            cv2.imwrite(filename0,image0)
                            ret, image2 = cam.read()
                            cv2.imwrite(filename1,image2)
                            print("***ALARM***") 
                            for x in range(20):
                                pause=random.randrange(3,7)/10

                                relay.on() # This will turn relay on
                                servo.max()
                                #print(relay.is_active)
                                time.sleep(pause)
                                relay.off() # turn the relay off
                                servo.min()
                                servo.max()
                                
                                #print(relay.is_active)
                                time.sleep(pause)
                                print(now)   






