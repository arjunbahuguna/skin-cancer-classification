#This script normalizes dimensions of images across entire dataset

import cv2
import os
import sys

height_of_images=[]
width_of_images=[]

def find_avg_dim(folder):
 	#read files from folder
    	for filename in os.listdir(folder):
		img = cv2.imread(os.path.join(folder,filename),0)    #accepts grayscale input, remove 0 to take RBG input
		if img is not None:
        		#extract height and width
			height, width = img.shape			
			height_of_images.append(height)		
			width_of_images.append(width)

#calculate avg dimensions
avg_height=sum(height_of_images)/len(height_of_images)
avg_width=sum(width_of_images)/len(width_of_images)


def resize(folder): 
	#read files from folder
	for filename in os.listdir(folder):
		img = cv2.imread(os.path.join(folder,filename),0)
		if img is not None:
	       		#edit
			resized = cv2.resize(img, (avg_width, avg_height), interpolation = cv2.INTER_AREA)
	        	#output
			cv2.imwrite(os.path.join(folder,filename),resized)
