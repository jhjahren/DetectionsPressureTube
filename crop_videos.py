import numpy as np
import cv2
import argparse
import sys
import pathlib as Path
import os

parser = argparse.ArgumentParser()
parser.add_argument('--vpath', type=str, default='C:/Users/janhe/PhDwork/PressureDataAnalysis/code_shared/DetectionsPressureTube/YoloV5/yolov5/data/video/Vid2.mp4', help='videopath')
parser.add_argument('--outpath', type=str, default='C:/Users/janhe/PhDwork/PressureDataAnalysis/code_shared/DetectionsPressureTube/YoloV5/yolov5/data/video/Vid2cropped.mp4', help='video output path')
args = parser.parse_args()

if (args.vpath!=None):
    print("A text path was given")
    print(args.vpath)

path = args.vpath
outpath = args.outpath
print(path)
print(outpath)
# Open the video
cap = cv2.VideoCapture(path)

# Initialize frame counter
cnt = 0

# Some characteristics from the original video
w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)

# Here you can define your croping values
x,y,h,w = 0,600,600,1600


# output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(outpath, fourcc, fps, (w, h))


# Now we start
while(cap.isOpened()):
    ret, frame = cap.read()
    cnt += 1 # Counting frames

    # Avoid problems when video finish
    if ret==True:
        # Croping the frame
        crop_frame = frame[y:y+h, x:x+w]

        # Percentage
        xx = cnt *100/frames
        print(int(xx),'%')

        # Saving from the desired frames
        #if 15 <= cnt <= 90:
        #    out.write(crop_frame)

        # I see the answer now. Here you save all the video
        out.write(crop_frame)

        # Just to see the video in real time
        #cv2.imshow('frame',frame)
        #cv2.imshow('croped',crop_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


cap.release()
out.release()
cv2.destroyAllWindows()