# First import the library
import pyrealsense2 as rs
# import numpy as np
#from guided import *
#from vehicle import Vehicle
#import defines
from threading import Thread
import numpy as np
from scipy import ndimage
#import time
import cv2
#import socketio
import base64
#import os
#import defines
#import socketio
#from socket_client import *
#import imutils
#from bisect import bisect_right
#import matplotlib.pyplot as plt
# from PIL import Image
# from scipy.misc import toimage
#from subprocess import Popen, PIPE
import subprocess

class Point_distance:

    def __init__(self):
        #self.vehicle = vehicle
        #self.sio = socketio.Client()
        return None

        # self.vehicle = vehicle

    def create_thread(self):
        # Thread(target=self.calculate_min_distance_decimation).start()
        # if(defines.blob_use==True):
        # Thread(target=self.calculate_min_distance_median_blob).start()
        # else:
        Thread(target=self.get_mynt_data).start()

        # Thread(target=self.color_stream).start()

        return self

    def get_mynt_data(self):
    	process = subprocess.Popen(['./get_depth_with_region'],stdout=subprocess.PIPE,stdin=subprocess.PIPE)
    	process.stdin.write(b'1')
    	#process.communicate()[0]
    	process.stdin.close()
    	while True:
	    	output = process.stdout.readline()
	    	if output=='' and process.poll() is not None:
	    		#print(output)
	    		break
	    	if output:
	    		obs_distance = int(output.strip().decode("utf-8"))
	    		print(obs_distance*.001)
	    	#rc = process.poll()
	    	#return rc
    	#while True:

    def main_filter(self, frame):
        source = frame
        edge_mask = self.filter_edge(source)
        harris_mask = self.filter_corner(source)
        combined_mask = cv2.bitwise_or(edge_mask, harris_mask)

    def filter_edge(self, frame):
        source = frame
        # skinCrCbHist = np.zeros((256, 256, 1), dtype = "uint16")
        source = np.int16(source)
        # cv::Scharr(area->decimated_ir, area->scharr_x, CV_16S, 1, 0);
        scharr_x = cv2.Scharr(source, -1, 1, 0)
        abs_scharr_x = cv2.convertScaleAbs(scharr_x)
        # cv::convertScaleAbs(area->scharr_x, area->abs_scharr_x);
        # cv::Scharr(area->decimated_ir, area->scharr_y, CV_16S, 0, 1);
        scharr_y = cv2.Scharr(source, -1, 0, 1)
        abs_scharr_y = cv2.convertScaleAbs(scharr_y)
        # cv::convertScaleAbs(area->scharr_y, area->abs_scharr_y);
        edge_mask = cv2.addWeighted(abs_scharr_y, 0.5, abs_scharr_x, 0.5, 0)
        # cv::addWeighted(area->abs_scharr_y, 0.5, area->abs_scharr_y, 0.5, 0, area->edge_mask);
        retval, binary = cv2.threshold(edge_mask, 192, 255, cv2.THRESH_BINARY)
        if(retval == True):
            return binary
        # cv::threshold(area->edge_mask, area->edge_mask, 192, 255, cv::THRESH_BINARY);

    def filter_corner(self, frame):
        source = frame
        float_ir = np.float32(source)
        # area->decimated_ir.convertTo(area->float_ir, CV_32F);
        corners = cv2.cornerHarris(float_ir, 2, 3, 0.04)
        # cv::cornerHarris(area->float_ir, area->corners, 2, 3, 0.04);
        retval, binary = cv2.threshold(corners, 300, 255, cv2.THRESH_BINARY)
        # cv::threshold(area->corners, area->corners, 300, 255, cv::THRESH_BINARY);
        if(retval == True):
            harris_mask = np.uint(binary)
        # area->corners.convertTo(area->harris_mask, CV_8U);

  
	

pint = Point_distance()
pint.create_thread()
