# MagangH8Kel7Day8_MAthallahY
This repo contains the files from Day 8

What is in this repo:
1. yolo_detect6.py          (the main python program)
2. PictureTest.jpg          (a sample picture)
3. my_model.pt              (the object detection model, required to run the main python program)

Example to run the main python program:
Wireless:
  python3 "yolo_detect6.py" --model my_model.pt --source PictureTest.jpg --thresh 0.5 --camAngle 76 --camLen 150 --pixScale 5.1333 --trapezType MP --commMethod wifi  --espIP http://192.168.1.10 

Serial:
  python3 "yolo_detect6.py" --model my_model.pt --source PictureTest.jpg --thresh 0.5 --camAngle 76 --camLen 150 --pixScale 5.1333 --trapezType MP --commMethod serial --serialPort COM7 --baudRate 9600
