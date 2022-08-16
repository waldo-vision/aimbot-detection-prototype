"""
predict.py
Usage: python predict.py {Input Video} {(optional) destination folder}

Takes in an input video file and runs it through the entire process and prdiction pipeline

Video file must be 1920 x 1080 @ 60FPS
Video should contin gameplay of Counter Strike: Global Offensive
"""

import os
import sys
import random
import subprocess


# Ensure that there is an input file
assert len(sys.argv) >= 2, "Requires path to input file"
inFile = sys.argv[1]
assert os.path.isfile(inFile), f'{inFile} isn\'t a valid file path'

dir = "temp"
delDir = True

# allow a specified output directory
if len(sys.argv) > 2:
    dir = sys.argv[2]
    assert os.path.isdir(dir), f'{dir} isn\'t a valid dirctory'
    assert len(os.listdir(dir)) == 0, "folder must be empty"
    delDir = False

#  Make a new unique folder if none is specified
if delDir:
    while os.path.isdir(dir):
        dir = "temp" + str(random.randint(1,1000000))
    os.mkdir(dir)

# Run auto clip to get clips in folder
print(f'python auto_clip.py \'{inFile}\' \'{dir}\' 1')
subprocess.run(['python', './auto_clip.py', inFile, dir, '1'])

# Process clips from auto_clip with cnn to extract features
print('\n\n\n\n\nNow extracting features')
subprocess.run(['python', 'save_cnn_output.py', dir, dir])

# Run extracted features through rnn to classify
print('\n\n\n\n\nNow Predicting value of cheats')
subprocess.run(['python', 'test_rnn.py', os.path.join(dir, 'clips.pt'), dir])


