# Import Module
import os
import re
from itertools import groupby
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='C:/Users/janhe/PhDwork/PressureDataAnalysis/code_shared/DetectionsPressureTube/YoloV5/yolov5/runs/detect/', help='videopath')
args = parser.parse_args()


# Folder Path
path = args.path
# Change the directory
os.chdir(path)
print('programmes are running again')

def read_text_file(file_path):
    with open(file_path, 'r') as f:
        print(f.read())

def write_text_file(file_path):
    with open(file_path,'r') as f:
        print(f.read())

def keyfunc(s):
    return [int(''.join(g)) if k else int(''.join(g)) for k, g in groupby(s, str.isdigit)]

# iterate through all file
frame = -1
file_lab = open('Labels.txt', 'w')
for file in sorted(os.listdir(), key=len):
    frame +=1
    # Check whether file is in text format or not
    if file.endswith(".txt"):
        file_path = f"{path}\{file}"
        file1 = open(file_path,'r')
        Lines = file1.readlines()
        for line in Lines:
            file_lab.write(str(frame) + " " + line + "\n")
        # call read text file function
file_lab.close()

    #read_text_file('Labels.txt')
