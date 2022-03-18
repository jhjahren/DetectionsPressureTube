import read_txtdata
import numpy as np
import argparse as args
import pathlib as Path
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from rdp import rdp
import math

ROOT = 'C:/Users/janhe/PhDwork/PressureDataAnalysis/code_shared/DetectionsPressureTube/YoloV5/yolov5'


def load_detections(pathstring):
    array = np.loadtxt(pathstring, delimiter=' ', converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None, like=None)
    print("The full dataset is of the shape: ",array.shape)
    return array


def extract_class_traj(data, classlabel):
    return None

def calculate_traj_length(traj):
    distance = 0
    for i in range(traj.shape[1]-1):
        x2,y2,x1,y1 = traj[1,i],traj[1,i],traj[0,i-1],traj[0,i-1]
        distance += math.hypot(x2-x1,y2-y1)
    return distance

parser = args.ArgumentParser()
parser.add_argument('--datapath', type=str, default=ROOT + '/Labels.txt', help='Path for the labels.txt data file of detections')
args = parser.parse_args()

path = ROOT + args.datapath
path = 'C:/Users/janhe/PhDwork/PressureDataAnalysis/code_shared/DetectionsPressureTube/YoloV5/yolov5/runs/detect/Vid80/labels/Labels.txt'
print(path)
data = load_detections(path)

df = pd.DataFrame(data)
df.columns = ['Frame','Class','x','y','w','h']
is_6, is_3 = df["Class"]==6, df["Class"]==3
df_head, df_tail = df[is_6], df[is_3]
df_head_coord, df_tail_coord = df_head[["Frame","x","y"]],df_tail[["Frame","x","y"]]
#print(df_head_coord.head(10))
print(df_head.tail())

merged_df = df_head_coord.merge(df_tail_coord, how = 'inner', on = ['Frame'])
print(merged_df.tail(10))

def angle(directions):
    """Return the angle between vectors
    """
    vec2 = directions[1:]
    vec1 = directions[:-1]

    norm1 = np.sqrt((vec1 ** 2).sum(axis=1))
    norm2 = np.sqrt((vec2 ** 2).sum(axis=1))
    cos = (vec1 * vec2).sum(axis=1) / (norm1 * norm2)
    return np.arccos(cos)



x,y = merged_df["x_x"], merged_df["x_y"]
t1,t2,t3,t4,t5,t6 = 8100, 9000, 9900, 10800, 11700, 12600

df1 = df_head[df_head['Frame'].between(t1, t2)]
df2 = df_head[df_head['Frame'].between(t2, t3)]
df3 = df_head[df_head['Frame'].between(t3, t4)]
df4 = df_head[df_head['Frame'].between(t4, t5)]
df5 = df_head[df_head['Frame'].between(t5, t6)]
print('Filtered traj nr1: \n',df1.tail(10))

x1,y1 = df1["x"],df1["y"]
x2,y2 = df2["x"],df2["y"]
x3,y3 = df3["x"],df3["y"]
x4,y4 = df4["x"],df4["y"]
x5,y5 = df5["x"],df5["y"]

x1, y1 = medfilt(x1, kernel_size= 75), medfilt(y1, kernel_size=75)
x2, y2 = medfilt(x2, kernel_size= 75), medfilt(y2, kernel_size=75)
x3, y3 = medfilt(x3, kernel_size= 75), medfilt(y3, kernel_size=75)
x4, y4 = medfilt(x4, kernel_size= 75), medfilt(y4, kernel_size=75)
x5, y5 = medfilt(x5, kernel_size= 75), medfilt(y5, kernel_size=75)


x1,y1 = x1.T, y1.T
trajectory1 = np.array((x1,y1))
x2,y2 = x2.T, y2.T
trajectory2 = np.array((x2,y2))
x3,y3 = x3.T, y3.T
trajectory3 = np.array((x3,y3))
x4,y4 = x4.T, y4.T
trajectory4 = np.array((x4,y4))
x5,y5 = x5.T, y5.T
trajectory5 = np.array((x5,y5))

print("The length of the trajectory 1 is:",calculate_traj_length(trajectory1))
print("The length of the trajectory 2 is:",calculate_traj_length(trajectory2))
print("The length of the trajectory 3 is:",calculate_traj_length(trajectory3))
print("The length of the trajectory 4 is:",calculate_traj_length(trajectory4))
print("The length of the trajectory 5 is:",calculate_traj_length(trajectory5))




fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x1, y1, 'r--', label='trajectory')
plt.title('Trajectory 1 - 4.5-5 min')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend(loc='best')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x2, y2, 'r--', label='trajectory')
plt.title('Trajectory 2 - 5-5.5 min')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend(loc='best')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x3, y3, 'r--', label='trajectory')
plt.title('Trajectory 3 - 5.5-6 min')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend(loc='best')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x4, y4, 'r--', label='trajectory')
plt.title('Trajectory 4 - 6-6.5 min')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend(loc='best')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x5, y5, 'r--', label='trajectory')
plt.title('Trajectory 5 - 6.5-7 min')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend(loc='best')
plt.show()

