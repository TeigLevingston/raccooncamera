# raccooncamera
An exercise in using Machine Learning on the edge using a Raspberry PI .<br/><br/>
TL;DR: I created a raccoon alarm using a Pi5, YoloV8, and some parts I had lying around to scare raccoons away from our cat bowl.
There are many tutorials, all of which are worthwhile and work, but they aren’t realistic applications. My purpose for the project was to gain more experience in building models and curating datasets.<br/><br/>
<b>Challenge</b><br/>
We live on a small farm in Texas with a barn cat. Well, she is pretty spoiled and lazy, so she is more of a porch cat. Her bowl sits on a counter on the porch, and lately, raccoons have been raiding it. A motion sensor was out of the question since having zero false positives was a requirement imposed by the Product Owner.<br/><br/>
<b>Equipment used:</b><br/>
A Raspberry PI 5 with an M.2 SSD (I don’t use SDs anymore)<br/>
5V One Channel Relay Module Relay Switch<br/>
DC 3-24V 95 dB Electronic Buzzer Active Piezo Buzzer Alarm Sounder Continuous Sound Beep, Black with a 9-volt battery for power<br/>
HXT 900 Servo (repurposed from an old RC plane)<br/>
Logitech USB web camera<br/><br/>
<b>Wiring</b><br/>
The wiring is what you would expect. The relay plugs into the Pi's 5V, ground, and pin 14. The Servo plugs into the Pi’s 5V, ground, and pin 24. I am using the gpiozero library with the default pin factory, though I may change it later to reduce the servo's PWM jitter. THe USB Webcam plugs into a USB Port.<br/><br/>
<b>Code</b><br/>
The code for the Pi is in the Pi folder. It is straightforward. I capture an image from the webcam, store it, and then capture a second image, comparing their numpy means to detect a change. When I detect a change, the image is processed through YoloV8 and the custom classification model I created with images of raccoons, cats, and various other things. The code creates a couple of files and pumps data to them for analysis.<br/>
The most interesting part of the project was creating the classification model. There are a number of generic models you can use for classification, but none of the ones I found included raccoons eating from a cat bowl as types. Using Yolo is an easy way to create a model, but unless the images are very representative, the model won’t function in the real world.<br/>
I started by trying to use the COCO data set supplemented with raccoon images from another source, but that failed horribly. By then, I had just enough images from the security camera I posted over the bowl to refine the dataset. Since I am only interested in classification, the images are organized into folders for the objects I am interested in.<br/>
With the new model, I could capture enough images of raccoons, cats, etc., to build a representative dataset. Since scaring the cat was an absolute no-go, false alarms were a major concern, the set probably won’t generalize. <br/>
The code for training the model using the YOLO library is straightforward. Several parameters can be passed through YOLO to configure the training set. I limited the image manipulation during training as the target environment is fixed. This will lead to a generalization limit of the model but expose more of the data curation elements I was interested in.
