import numpy as np
import cv2
import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import read_txtdata

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

parser = argparse.ArgumentParser()
parser.add_argument('--vpath', type=str, default=ROOT / 'labels.txt', help='videopath')
parser.add_argument("--txtpath", type=str, default=ROOT / 'vid.avi', help='txtpath')
args = parser.parse_args()

if (args.txtpath!=None):
    print("A text path was given")
    print(args.vpath)
    print(args.txtpath)



cap = cv2.VideoCapture(args.vpath)
ret, frame = cap.read()

height,width = frame.shape[0], frame.shape[1]

array = read_txtdata.load_detections(args.txtpath)
eye_array = read_txtdata.filter_class(array,1)
eye_array_frame1 = read_txtdata.filter_frame(eye_array,1)
eye_coordinates = read_txtdata.yolo_to_coordinates(eye_array_frame1, width, height)

# params for ShiTomasi corner detection
# throw every other corners below quality level. Sort rest in descending order. Pick greatest, throw rest in min and pick N greatest
feature_params = dict( maxCorners = 100,      # how many pts. to locate
                       qualityLevel = 0.3,  # b/w 0 & 1, min. quality below which everyone is rejected
                       minDistance = 1,     # min eucledian distance b/w corners detected
                       blockSize = 7 )      #

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),   # size of the search window at each pyramid level
                  maxLevel = 2,       #  0, pyramids are not used (single level), if set to 1, two levels are used, and so on
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                  # Criteria : Termination criteria for iterative search algorithm.
                  # after maxcount { Criteria_Count } : no. of max iterations.
                  # or after { Criteria Epsilon } : search window moves by less than this epsilon


# Create some random color for the pt. chosen
color = np.random.randint(0,255,(1,3))

# Take first frame and find corners in it

ret, old_frame = cap.read()     #read frame


old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)  #convert to grayscale

p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)   #use goodFeaturesToTrack to find the location of the good corner.

print("The datatype of p0 is: ", p0.dtype)
p0 = np.float32(eye_coordinates)
#cvPoint pl = new cvPoint(2,3)

print(p0.shape)
print("The datatype of p0 is: ", p0.dtype)

# Create a mask image for drawing purposes filed with zeros
mask = np.zeros_like(old_frame)
while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    # err kind of gives us the correlation error(matching error)

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()

        mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), (255,0,0), 2)
        frame = cv2.circle(frame,(int(a),int(b)),5,(0,255,0),-1)
    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
#
#    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
#
## release and destroy all windows.
cv2.destroyAllWindows()
cv2.release()