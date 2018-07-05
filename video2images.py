#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
import sys

#Simple function toconvert video to image frames


def _main_():
 
    #define paths e.g.
    # /media/ubuntu/hdd/tensorflow_data/YOLO/CowData/train_images/cow-2.jpg
    video_path = '/media/ubuntu/hdd/tensorflow_data/YOLO/PeopleData/campuspeople.mp4'
    image_folder   = '/media/ubuntu/hdd/tensorflow_data/YOLO/PeopleData/train_images2/'
    image_root = 'people_'

    index_offset = 541 #0

    #desired image size
    WIDTH = 640
    HEIGHT = 480

    #open up video
    video_reader = cv2.VideoCapture(video_path)

    #determine number of frames
    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    print('Starting Frame Conversion')
    for i in range(nb_frames):
        #grab current frame
        _, image = video_reader.read()

        image=cv2.resize(image, (WIDTH, HEIGHT)) 
        
        #make filename
        filetemp = image_folder + image_root + str(i + index_offset) + '.jpeg'

        #write file
        cv2.imwrite(filetemp, image)

        #percent complete
        percent = (float(i) / float(nb_frames)) * 100.0
        print "%.0f %% Complete"%percent
        
    
    #release
    video_reader.release()

    #feedback
    print 'Processed %i frames'%(nb_frames)
        
        
if __name__ == '__main__':
    _main_()
