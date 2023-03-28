import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import yaml
import csv


arr=[]

with open('shift_analysis.yaml') as Config_file:    
    analysis_config = yaml.load(Config_file, Loader=yaml.FullLoader)
        
input_directory=analysis_config['inputpath']
image_format=analysis_config['image_format']
kernel_size = analysis_config['Guassian_blur']['kernel_size']

low_threshold = analysis_config['Canny_edges']['low_threshold']
high_threshold = analysis_config['Canny_edges']['high_threshold']

rho=analysis_config['HoughLine_Params']['rho']
threshold=analysis_config['HoughLine_Params']['threshold']
minLineLength=analysis_config['HoughLine_Params']['minLineLength']
maxLineGap=analysis_config['HoughLine_Params']['maxLineGap']

color=(255,0,0)
thickness=12
# input_directory="/jidoka/workspace/jdisk2/Krishna/trigger_exp/single_ob1/VCXU-32C/VCXU-32C_700008191180_221226-152329/"
file_names=os.listdir(input_directory)
for filename in file_names:
	if image_format in filename:
		# print(filename)
		input_path=os.path.join(input_directory,filename)
		img=cv2.imread(input_path)
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
		edges = cv2.Canny(blur, low_threshold, high_threshold)
		lines = cv2.HoughLinesP(edges,rho,np.pi/180,threshold=threshold,minLineLength=minLineLength,maxLineGap=maxLineGap)
		for points in lines:
			x1,y1,x2,y2=points[0]
			try:
				if x1==x2:
					if x1 <1000:
						cv2.line(img,(x1,0),(x2,1536), color, thickness)
						shift_x = abs(549 - x1)
						arr.append(['single_ob1',filename,549,x1,shift_x])
						break
			except Exception as e:
				pass
print(len(arr))
				


