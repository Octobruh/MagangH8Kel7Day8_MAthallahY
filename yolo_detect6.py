import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
import math as mt
import serial
import requests  # Added for WiFi communication
from ultralytics import YOLO

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file', required=True)
parser.add_argument('--source', help='Image source (file, folder, video, "usb0", or "picamera0")', required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold', default=0.5)
parser.add_argument('--camAngle', help='Angle of camera measured from horizontal in degrees', default=90)
parser.add_argument('--camLen', help='height of camera from ground in cm', default=150)
parser.add_argument('--pixScale', help='Pixel Scale = Pixels/Len<cm>', default=5.1385)
parser.add_argument('--trapezType', help='Bottom Anchor: <BA>, Middle Point: <MP>, Bottom Anchor No Foreshort: <BT>', choices=['BA', 'BT', 'MP'], default='BA')

parser.add_argument('--commMethod', help='Communication method', choices=['serial', 'wifi', 'none'], default='none')
parser.add_argument('--espIP', help='IP address for WiFi communication (e.g., http://192.168.1.10)', default='http://192.168.1.10')
parser.add_argument('--serialPort', help='Serial port (e.g., COM7 or /dev/ttyUSB0)', default='COM7')
parser.add_argument('--baudRate', help='Baud rate for serial communication', type=int, default=9600)

parser.add_argument('--resolution', help='Resolution (WxH). Note: This script will force 640x480 for the transform.', default='640x480') # Changed default
parser.add_argument('--record', help='Record results from video or webcam', action='store_true')

args = parser.parse_args()
# Example Run:
# python3 "yolo_detect6.py" --model my_model.pt --source PictureTest.jpg --thresh 0.5 --camAngle 76 --camLen 150 --pixScale 5.1333 --trapezType MP --commMethod wifi  --espIP http://192.168.1.10 
# python3 "yolo_detect6.py" --model my_model.pt --source PictureTest.jpg --thresh 0.5 --camAngle 76 --camLen 150 --pixScale 5.1333 --trapezType MP --commMethod serial --serialPort COM7 --baudRate 9600

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
cameraAngle = float(args.camAngle)
cameraLen = float(args.camLen)
pixlScale = float(args.pixScale)
trapezoidType = args.trapezType
user_res = args.resolution
record = args.record

# Parse communication arguments
comm_method = args.commMethod
esp_ip = args.espIP
port_a = args.serialPort
baud_rate = args.baudRate

# PixelScale = 5.1385    # Calculate at 90 degree angle
PixelScale = pixlScale

# theta_degrees = 82     # Angle in degrees
theta_degrees = cameraAngle
theta_radians = mt.radians(theta_degrees)

# IMPORTANT! These values are in pixels (px).
X_input = 640.0                 # Image resolution X from camera
Y_input = 480.0                 # Image resolution Y from camera
# L_input = 74.6 * PixelScale   # Camera height in PIXELS
L_input = cameraLen * PixelScale
H_input = 480.0                 # Ground area 'tallness'

# The following function calculates the trapezoid for the perspective shift.
def calculate_coordinates(X, Y, L, H, theta, trapezoidType):
    # Calculate sin(theta) and cos(theta)
    sin_theta = mt.sin(theta)
    cos_theta = mt.cos(theta)
    
    # Calculate the common fractional part to avoid repetition
    # (2L - H * sin(theta) * cos(theta)) / (2L + H * sin(theta) * cos(theta))
    try:
        numerator = 2 * L - H * sin_theta * cos_theta
        denominator = 2 * L + H * sin_theta * cos_theta
        
        if denominator == 0:
            print("Error: Division by zero. Denominator (2L + H*sin(theta)*cos(theta)) is zero.")
            return None
            
        common_factor = numerator / denominator
    except Exception as e:
        print(f"Error during calculation of common_factor: {e}")
        return None

    # Calculate Xt = 1/2 * [X +- X * common_factor]
    Xtl = 0.5 * (X - X * common_factor)
    Xtr = 0.5 * (X + X * common_factor)

    if trapezoidType == 'BA':
        # Calculate Yt = Y - (H * sin(theta))
        Ytl = Y - (H * sin_theta)
        Ytr = Y - (H * sin_theta)
        Ybl = Y
        Ybr = Y
    else:
        # Alternative, 8-11-2025 adjustment. Constant Y Middlepoint.
        # Instead of constantly anchored bottom of trapezoid to bottom of screen.
        Ytl = 0.5 * (Y - H * sin_theta)
        Ytr = 0.5 * (Y - H * sin_theta)
        Ybl = 0.5 * (Y + H * sin_theta)
        Ybr = 0.5 * (Y + H * sin_theta)

    # Return the results in a dictionary for clarity
    return {
        'Xtl': Xtl,
        'Ytl': Ytl,
        'Xtr': Xtr,
        'Ytr': Ytr,
        'Ybl': Ybl,
        'Ybr': Ybr
    }

def getBoundingCenters(xmin, xmax, ymin, ymax, centers_x, centers_y, centers_xR, centers_yR, Y_input, labeles, label, PixelScale):
    # Calculate center point
    center_x = int((xmin + xmax) / 2)
    center_y = int((ymin + ymax) / 2)
    
    # Draw circle at center point
    # Will pass 'transformed_frame' to this function, so it needs to be defined
    cv2.circle(transformed_frame, (center_x, center_y), 5, (0, 0, 255), -1)

    # Append to list of centers
    centers_x.append(center_x)
    centers_y.append(center_y)
    labeles.append(label)

    # Calculate real center location
    # PixelScale = 5.333333
    center_xR = center_x / PixelScale

    # Length and Width of bounding boxes to estimate object dimension in cm
    # Also useful to measure the geometric distortion of the object
    len_x = int(xmax - xmin) / PixelScale
    len_y = int(ymax - ymin) / PixelScale

    # Center_yR = center_y / PixelScale
    # This because I shifted the 0 position from top of image to bottom of image
    # Since this is symetric, I used absolute value of the result
    center_yR = abs(center_y - Y_input) / PixelScale

    centers_xR.append(center_xR)
    centers_yR.append(center_yR)

    return {
        'panjang': len_x,
        'lebar': len_y
    }

# Check if model file exists
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid.')
    sys.exit(0)

# Load the model
model = YOLO(model_path, task='detect')
labels = model.names

# List of centers
centers_x = []
centers_y = []
centers_xR = []
centers_yR = []
labeles = []

# Force the resolution for consistency with the transform
resW, resH = int(X_input), int(Y_input)
resize = True 

if user_res != f"{resW}x{resH}":
    print(f"WARNING: Overriding user-specified resolution '{user_res}' with transform resolution '{resW}x{resH}'.")

# Calculate transform coordinates
print("Calculating perspective transform matrix...")
coordinates = calculate_coordinates(X_input, Y_input, L_input, H_input, theta_radians, trapezoidType)
if coordinates is None:
    print("Error calculating transform coordinates. Exiting.")
    sys.exit(1)

if trapezoidType == 'BA':
    # Bottom anchor trapezoid
    tl = (int(coordinates['Xtl']), int(coordinates['Ytl']))
    bl = (int(0), int(Y_input))
    tr = (int(coordinates['Xtr']), int(coordinates['Ytr']))
    br = (int(X_input), int(Y_input))

elif trapezoidType == 'BT':
    # Bottom anchor trapezoid, no top foreshortening
    tl = (int(coordinates['Xtl']), 0)
    bl = (int(0), int(Y_input))
    tr = (int(coordinates['Xtr']), 0)
    br = (int(X_input), int(Y_input))

elif trapezoidType == 'MP':
    # Middle point trapezoid
    tl = (int(coordinates['Xtl']), int(coordinates['Ytl']))
    bl = (int(0), int(coordinates['Ybl']))
    tr = (int(coordinates['Xtr']), int(coordinates['Ytr']))
    br = (int(X_input), int(coordinates['Ybr']))

# Define transform points
pts1 = np.float32([tl, bl, tr, br]) # Trapezoid corners
pts2 = np.float32([[0,0], [0,Y_input], [X_input,0], [X_input,Y_input]]) # Original image corners

# Calculate the transformation matrix ONCE
transform_matrix = cv2.getPerspectiveTransform(pts1, pts2)
print("Transform matrix calculated successfully.")

# Parse input to determine source type
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
elif 'http' in img_source:
    source_type = 'video'
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Check if recording is valid
if record:
    if source_type not in ['video','usb']:
        print('Recording only works for video and camera sources.')
        sys.exit(0)
    # The 'if not user_res' check is no longer needed since we force it
    
    # Set up recording
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb':
    if source_type == 'video': cap_arg = img_source
    elif source_type == 'usb': cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)
    # Set camera or video resolution
    cap.set(3, resW)
    cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()

# Set bounding box colors
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
             (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Define transformed_frame here to be in scope for getBoundingCenters
transformed_frame = None

print("Starting inference loop...")
# Begin inference loop
k = 0 # To index multiple image capture from camera
while True:

    t_start = time.perf_counter()

    # Load frame from image source
    if source_type == 'image' or source_type == 'folder':
        if img_count >= len(imgs_list):
            # When done, print the coordinate results
            print('All images have been processed. Exiting program.')
            for i in range(len(centers_x)):
                print(centers_x[i], ",", centers_y[i])
            print("--------------Real Coordinates-------------")
            for i in range(len(centers_xR)):
                print(labeles[i], round(centers_xR[i], 2), ",", round(centers_yR[i], 2))
            
            # Sending Data
            if comm_method == 'serial':
                try:
                    with serial.Serial(port_a, baud_rate, timeout=2) as ser:
                        ser.flush()
                        print(f"Main script connected to {ser.name} at {baud_rate} baud")
                        print("Main script sending object data (serial)...")
                        
                        for i in range(1):
                            labeln = labeles[i]
                            x_coord = round(centers_xR[i], 2)
                            y_coord = round(centers_yR[i], 2)
                            
                            message = f"{labeln},{x_coord},{y_coord}\n"
                            # message = f"{x_coord}\n"
                            
                            print(f"Sending (serial): {message.strip()}")
                            ser.write(message.encode('utf-8'))
                            time.sleep(2.0) 

                        # print("Sending (serial): DONE")
                        # ser.write(b"DONE\n")

                        response = ser.readline()
                        
                        if response:
                            print(f"Main script received: {response.decode('utf-8').strip()}")
                        else:
                            print("Main script: No response from serial device.")

                except serial.SerialException as e:
                    print(f"Main script serial error: {e}")
            
            elif comm_method == 'wifi':
                try:
                    print(f"Main script sending object data (WiFi) to {esp_ip}...")
                    
                    for i in range(len(labeles)):
                        labeln = labeles[i]
                        x_coord = round(centers_xR[i], 2)
                        y_coord = round(centers_yR[i], 2)
                        
                        # Format from demo.py: "Label: X, Y"
                        message = f"{labeln}: {x_coord}, {y_coord}"
                        
                        r = requests.post(f"{esp_ip}/data", data={"msg": message})
                        print(f"Sending (wifi): '{message}' -> {r.text}")

                    # After loop, send the /run command
                    print("Sending (wifi): RUN")
                    r = requests.get(f"{esp_ip}/run")
                    print(f"Main script received: {r.text}")

                except requests.exceptions.RequestException as e:
                    print(f"Main script WiFi error: {e}")
            
            elif comm_method == 'none':
                print("Communication method set to 'none'. Skipping data sending.")
            
            sys.exit(0)
            
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count = img_count + 1
    
    elif source_type == 'video':
        ret, frame = cap.read()
        if not ret:
            print('Reached end of the video file. Exiting program.')
            break
    
    elif source_type == 'usb':
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print('Unable to read frames from the camera. Exiting.')
            break

    elif source_type == 'picamera':
        frame = cap.capture_array()
        if (frame is None):
            print('Unable to read frames from the Picamera. Exiting.')
            break

    # 1. Resize frame to the required 640x480 resolution
    if resize == True:
        frame_resized = cv2.resize(frame, (resW, resH))
    else:
        frame_resized = frame

    # 2. Apply the perspective transform
    transformed_frame = cv2.warpPerspective(frame_resized, transform_matrix, (resW, resH))

    # Draw the trapezoid for visualisaton
    cv2.circle(frame_resized, tl, 5, (0,0,255), -1)
    cv2.circle(frame_resized, bl, 5, (0,0,255), -1)
    cv2.circle(frame_resized, tr, 5, (0,0,255), -1)
    cv2.circle(frame_resized, br, 5, (0,0,255), -1)

    cv2.line(frame_resized, tl, tr, (0,0,255), 2)
    cv2.line(frame_resized, tr, br, (0,0,255), 2)
    cv2.line(frame_resized, br, bl, (0,0,255), 2)
    cv2.line(frame_resized, bl, tl, (0,0,255), 2)

    # 3. Run YOLO inference on the TRANSFORMED frame
    results = model(transformed_frame, verbose=False) # Pass transformed_frame here

    # Extract results
    detections = results[0].boxes
    object_count = 0

    # Go through each detection
    for i in range(len(detections)):
        # Get bounding box coordinates
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        # Get class ID and name
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        # While inferencing, print length and width in pixels, this to help with calibration
        print(f"{classname} len_x: {(xmax - xmin)/PixelScale}, len_y: {(ymax - ymin)/PixelScale}, Pixel")

        # Get confidence
        conf = detections[i].conf.item()

        # Draw box if confidence is high enough
        if conf > min_thresh:
            color = bbox_colors[classidx % 10]
            # Draw on the transformed_frame
            cv2.rectangle(transformed_frame, (xmin,ymin), (xmax,ymax), color, 2)

            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(transformed_frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(transformed_frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            labelname = str(classname)

            # Get bounding centers (this will now use transformed_frame)
            getBoundingCenters(xmin, xmax, ymin, ymax, centers_x, centers_y, centers_xR, centers_yR, Y_input, labeles, labelname, PixelScale)
            
            object_count = object_count + 1

    # Calculate and draw framerate
    if source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        cv2.putText(transformed_frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    
    # Display detection results on the transformed frame
    cv2.putText(transformed_frame, f'Number of objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    cv2.imshow('YOLO detection results (Transformed)', transformed_frame) # Display the transformed frame
    
    # Optional: Show the original, resized frame for comparison
    cv2.imshow('Original (Resized)', frame_resized)
    
    if record: 
        recorder.write(transformed_frame) # Record the transformed frame

    # Handle keypresses
    if source_type == 'image' or source_type == 'folder':
        key = cv2.waitKey()
    elif source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        key = cv2.waitKey(5)
    
    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord('s') or key == ord('S'):
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'):
        cv2.imwrite("capture_transformed" + str(k) + ".png", transformed_frame)
        cv2.imwrite("capture_original" + str(k) + ".png", frame_resized)
        print(f"Saved frames as capture...{k}.png")
        k = k + 1
    
    # Calculate FPS
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
        
    frame_rate_buffer.append(frame_rate_calc)
    
    avg_frame_rate = np.mean(frame_rate_buffer)

# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type == 'video' or source_type == 'usb':
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record: 
    recorder.release()
cv2.destroyAllWindows()

# TODO :
# Real Testing on real environment