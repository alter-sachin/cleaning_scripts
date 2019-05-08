# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

extensionsToCheck = ['.txt', '.py', '.dat']
#if any(ext in url_string for ext in extensionsToCheck):
#    print(url_string)
folders = os.listdir()
print(folders)
for folder in folders:
	if any(ext in str(folder) for ext in extensionsToCheck):
		continue
	else:	
		os.chdir(str(folder))
		imagesx = os.listdir()
		c = 0
		for image in imagesx:
			if ".png" in image:
				image1 = cv2.imread(str(image))
				gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

				# detect faces in the grayscale image
				rects = detector(gray, 1)
				#c = 0
				# loop over the face detections
				for (i, rect) in enumerate(rects):
					# determine the facial landmarks for the face region, then
					# convert the facial landmark (x, y)-coordinates to a NumPy
					# array
					shape = predictor(gray, rect)
					shape = face_utils.shape_to_np(shape)
					# convert dlib's rectangle to a OpenCV-style bounding box
					# [i.e., (x, y, w, h)], then draw the face bounding box
					(x, y, w, h) = face_utils.rect_to_bb(rect)
					print(x,y,w,h)
					
					if x > 0 :
						#seed = np.random(1000)
						try :	
							c = c +1
							if c == 1:
								os.makedirs("../results/"+str(folder))
							cv2.imwrite("../results/"+str(folder)+"/"+str(image),image1)
						except Exception as e:
							print(e)
	os.chdir("..")				
