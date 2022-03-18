import numpy as np
import cv2 as cv
import argparse as args


def get_detected_eyes():
    lines = []
    with open(args.txtpath,'r') as file:
        lines = file.readlines()
        for line in lines:
            print(line)


def load_detections(pathstring):
    array = np.loadtxt(pathstring, delimiter=' ', converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None, like=None)
    print("The full dataset is of the shape: ",array.shape)
    return array


def filter_class(array, classlabel):
    new_array = array[array[:,1] == classlabel]
    print("The class filtered data has the shape: ", new_array.shape)
    return new_array

def filter_frame(array, framenr):
    new_array = array[array[:,0] == framenr]
    print("The frame filtered data has the shape: ",new_array.shape)
    return new_array

def yolo_to_coordinates(a,x, y):
    coords = a[:,[2,3]]
    coords[:,[0]] = coords[:,[0]]*x
    coords[:,[1]] = coords[:,[1]]*y
    xy = coords[:,np.newaxis,:]
    print("The coordinates have the shape:", xy.shape)
    return xy



