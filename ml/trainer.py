#This code runs om my ML desktop. I tried building the model on the Pi but it failed, I don't think there was enough memory. I was able to build the model on an ubuntu server with 32GB of RAM.
if __name__ == '__main__':
    from ultralytics import YOLO
    from datetime import datetime
# Load a model
    model = YOLO("yolov8m-cls.pt")  # load a pretrained model (recommended for training)
#Load empty model
#model = YOLO("")
    model.to('cuda')
# Train the model
    startTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    results = model.train(data="datasets\\cococat2", epochs=30, imgsz=640, save=True, batch = 16, task='classify',augment=False,auto_augment=None,translate=0, scale=0, fliplr=0, erasing=0,crop_fraction=1)

    endTime =   datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print ("start: "   + startTime)
    print ("End Time: " + endTime)
