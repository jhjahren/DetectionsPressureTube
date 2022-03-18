import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import argparse as args
import pandas as pd


def load_detections(pathstring):
    array = np.loadtxt(pathstring, delimiter=' ', converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None, like=None)
    print("The full dataset is of the shape: ",array.shape)
    return array


def extract_class_traj(data, classlabel):
    return None

def crop_frame(frame,x,y,w,h):
    out = frame[round(y-h/2):round(y+h/2),round(x-w/2):round(x+w/2)]
    return out

def yolo_to_pixelcoords(frame,x,y,w,h):
    frame_h, frame_w, channels = frame.shape
    x_out, y_out, w_out, h_out = x*frame_w,y*frame_h,w*frame_w,h*frame_h
    return round(x_out), round(y_out), round(w_out), round(h_out)


parser = args.ArgumentParser()
parser.add_argument('--datapath', type=str, default=None, help='Path for the labels.txt data file of detections')
parser.add_argument('--vidpath', type=str, default=None, help='Video path')
args = parser.parse_args()

dpath = args.datapath
vpath = args.vidpath

df = pd.DataFrame(load_detections(dpath))
df.columns = ['Frame','Class','x','y','w','h','Conf']
is_6 = df["Class"]==3
df_head = df[is_6]
is_confident = df["Conf"]>=0.8
filtered_df = df_head[is_confident]
print(df.head(20))
#Thresholding at pixel values below 70 seems good
threshold = 65

# Creating a VideoCapture object to read the video
cap = cv2.VideoCapture(vpath)

cnt = 1
white_pixels = []
x,y,w,h = 600,300,200,200
# Loop until the end of the video
color = (0, 0, 255)
while (cap.isOpened()):


    # Capture frame-by-frame
    ret, frame = cap.read()
    #frame = cv2.resize(frame, (540, 380), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)


    # values on different regions of the frame.
    frame = cv2.rectangle(frame, (round(x - w / 2), round(y - h / 2)), (round(x + w / 2), round(y + h / 2)), color, 2)
    cv2.imshow('Frame', frame)

    color = (0,0,255)
    #update coordinates if a detection is made:
    if cnt in filtered_df["Frame"].values:
        # Display the resulting frame
        color = (0,255,0)

        # conversion of BGR to grayscale is necessary to apply this operation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_df = df_head["Frame"]==cnt
        det = df_head[frame_df]
        detvalues = det.values
        detX, detY = detvalues[0,2], detvalues[0,3]
        print("The count is:", cnt)
        print("Det is then:\n",det)
        print("The detected x and y coordinates: \n", detvalues[0,2], "and ", detvalues[0,3])

        # Crop frames
        x,y,w,h = yolo_to_pixelcoords(frame,detX,detY,0.15,0.2)
        gray_cropped = crop_frame(gray,x,y,w,h)



        ret, thresh_img = cv2.threshold(gray_cropped, threshold, 255, cv2.THRESH_BINARY_INV)

        try:
            cv2.imshow('Thresh', thresh_img)
        except Exception as e:
            pass

        #count white pixels
        white_pixel_count = np.sum(thresh_img == 255)
        print(white_pixel_count)
        white_pixels.append(white_pixel_count)


    # define q as the exit button
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    if cnt > 1500:
        break
    cnt += 1

# release the video capture object
cap.release()
# Closes all the windows currently opened.
cv2.destroyAllWindows()

white_pixels = np.array(white_pixels)
x = np.arange(1,len(white_pixels)+1,1)

plt.plot(x,white_pixels)
plt.show()

f, pxx =  signal.periodogram(white_pixels,1)
plt.plot(f,pxx)
plt.show()