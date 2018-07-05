#! /usr/bin/env python

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" #"0" = use GPU, "" = use CPU


import argparse
import cv2
import numpy as np
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import json
import time


def _main_():
 
    #where are weights and video are
    weights_path = '/media/ubuntu/hdd/tensorflow_data/YOLO/PeopleData/yolo_people_final.h5'
    video_path   = '/media/ubuntu/hdd/tensorflow_data/YOLO/PeopleData/campuspeople.mp4'
    video_out   = '/media/ubuntu/hdd/tensorflow_data/YOLO/PeopleData/campuspeople_detected.mp4'
    RECORD = False

    #Configuration
    architecture = "Full Yolo"
    input_size = 416
    #anchors = [0.56,2.23, 0.83,3.81, 1.40,6.19, 2.20,9.83, 4.04,10.64]
    anchors = [0.58,0.97, 1.09,1.84, 1.79,5.82, 2.69,9.29, 5.14,10.71]
    max_box_per_image = 10
    labels = ["person"]

    #image resize
    WIDTH = 640
    HEIGHT = 480

    ###############################
    #   Make the model 
    ###############################

    model = YOLO(architecture        = architecture,
                input_size          = input_size, 
                labels              = labels, 
                max_box_per_image   = max_box_per_image,
                anchors             = anchors,
                pretrained_weights  = '/media/ubuntu/hdd/tensorflow_data/YOLO/full_yolo_backend.h5')

    ###############################
    #   Load trained weights
    ###############################    

    print("loading weights from")
    print(weights_path)
    model.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################

    print("opening video:")
    print(video_path)
    video_reader = cv2.VideoCapture(video_path)

    #video properties
    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(video_reader.get(cv2.CAP_PROP_FPS))
    print("number of frames:")
    print(nb_frames)
    print("FPS:")
    print(fps)

    #window to display images
    cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
    cv2.startWindowThread() #to make sure we can close it later on

    if(RECORD):
        #video writer
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_writer = cv2.VideoWriter(video_out, fourcc, fps, (WIDTH,HEIGHT))

    meantime = 0.0
    index = 0 
    for i in range(nb_frames):
        _, image = video_reader.read()

        image=cv2.resize(image, (WIDTH, HEIGHT)) 

        #predict objects
        t0 = time.time()
        boxes = model.predict(image)
        t1 = time.time()
        total = t1-t0
        meantime = meantime + total
        index = index + 1

        #draw objects
        image = draw_boxes(image, boxes, labels)

        #record
        if(RECORD):
            video_writer.write(image)

        
        #display frame
        image_np = np.uint8(image)
        cv2.imshow("Detection", image_np)
        k = cv2.waitKey(5) & 0xEFFFFF
        if k == 27: 
            print("You Pressed Escape")
            break


    meantime = meantime / index
    print("Detection Time %f s"%meantime)
    #release video
    video_reader.release()

    #destroy windows
    cv2.destroyAllWindows()
    for i in range (1,5):
        cv2.waitKey(1)
        
if __name__ == '__main__':
    _main_()
