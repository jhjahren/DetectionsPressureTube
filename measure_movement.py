import read_txtdata
import numpy as np
import argparse as args
import pathlib as Path
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import medfilt, periodogram
from rdp import rdp

ROOT = 'C:/Users/janhe/PhDwork/PressureDataAnalysis/code_shared/DetectionsPressureTube/YoloV5/yolov5'


def load_detections(pathstring):
    array = np.loadtxt(pathstring, delimiter=' ', converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None, like=None)
    print("The full dataset is of the shape: ",array.shape)
    return array


def extract_class_traj(data, classlabel):
    return None


parser = args.ArgumentParser()
parser.add_argument('--datapath', type=str, default=None, help='Path for the labels.txt data file of detections')
parser.add_argument('--vidnum', type=str, default=None, help='Videonumber')
args = parser.parse_args()

path = args.datapath
print(path)
data = load_detections(path)


#Insert framrate
framerate = 30
vidnum = args.vidnum

df = pd.DataFrame(data)
df.columns = ['Frame','Class','x','y','w','h']
is_6, is_3 = df["Class"]==6, df["Class"]==3
df_head, df_tail = df[is_6], df[is_3]
df_head_coord, df_tail_coord = df_head[["Frame","x"]],df_tail[["Frame","x"]]


num_frames, first_frame = df_head['Frame'].iloc[-1], df_head['Frame'].iloc[0]
num_frames, first_frame = int(num_frames), int(first_frame)
#plt.scatter(df['Frame']/framerate, df['x'])
#plt.show()

#plt.scatter(df['Frame']/framerate, df['y'])
#plt.show()

# Set the length equal to the frame number of the final detection
time = np.linspace(first_frame+1,num_frames-1,num=num_frames-first_frame,endpoint=True)
time_5 = (9000)/framerate
time_6 = (10800)/framerate



x = np.array(df_head['x'])
t = np.array(df_head['Frame'])
print(t.min(),t.max())
print(time.min(), time.max())
fx = interp1d(t,x,kind='linear')
fx2 = medfilt(fx(time), kernel_size= 31)

fig1 = plt.figure(figsize=(8,4))
plt.plot(time/framerate, fx(time), '-', time/framerate, fx2,'r--')
plt.axvline(x = time_5, color = 'g', lw = 3)
plt.axvline(x = time_6, color = 'g', lw = 3)
plt.legend(['linear x', 'smoothed x'], loc='best')
plt.savefig('Figures/X-trajectory'+vidnum)
#plt.show()

y = np.array(df_head['y'])
t = np.array(df_head['Frame'])
fy = interp1d(t,y,kind='linear')
fy2 = medfilt(fy(time), kernel_size= 5)

fig2 = plt.figure(figsize=(8,4))
fig2 = plt.plot(time/framerate, fy(time), '-', time/framerate, fy2,'r--')
plt.axvline(x = time_5, color = 'g', lw = 3)
plt.axvline(x = time_6, color = 'g', lw = 3)
plt.legend(['linear y', 'smoothed y'], loc='best')
plt.savefig('Figures/Y-trajectory'+vidnum)
#plt.show()



#Trying to calculate the speed
dx = 0.05
x = np.cumsum(np.abs(fx2))

# we calculate the derivative, with np.gradient
fig3 = plt.figure(figsize=(8,4))
plt.plot(time/framerate,medfilt(np.gradient(fx2, dx),kernel_size=31), '-y', label='Speed')
plt.plot(time/framerate, fx2,'r--')
plt.axvline(x = time_5, color = 'g', lw = 3)
plt.axvline(x = time_6, color = 'g', lw = 3)
plt.axhline(y = 0, color = 'g', lw = 3)
plt.legend(['speed', 'smoothed x'], loc='best')
plt.savefig('Figures/Speed'+vidnum)
#plt.show()

fig3= f, pxx = periodogram(medfilt(np.gradient(fx2, dx),kernel_size=31),1)
plt.plot(f,pxx)
plt.show()


print(x.shape)
print(time.shape)


