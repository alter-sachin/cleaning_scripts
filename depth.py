import cv2

import numpy as np
image = cv2.imread("example.jpg")
hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

lowerred = np.array([0,50,50])
upperblue = np.array([60,255,255])

mask = cv2.inRange(hsv,lowerred,upperblue)
cv2.imwrite("mask.jpg",mask)
#for x in range(0,423):
#	for y in range(0,239):
#		rgb_value = (image[y,x])
#		print(rgb_value)
		
