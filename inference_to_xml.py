############################### IMPORTS #####################################################
import json
import sys
sys.path.append("/jidoka/code/jt.prod.training/")
from utils import json_utils

import numpy as np
import os
import sys , pickle
sys.path.append('/jidoka/code/jt.prod.training/visualization')
sys.path.append('/jidoka/code/jt.prod.training/utils')
import errno
from PIL import Image
import boxes_utils as vis_util
import tensorflow as tf
from inference.infer_model_sess import *
import matplotlib.pyplot as plt
import json
import cv2

from file_utils import make_sure_path_exists
# %matplotlib inline
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#################################################################################################

LABELMAP_PATH = '/jidoka/simulations/jdisk2/jt.cust.Shriji_caps/data_object_detection/data_1/split_image/labelmap.pbtxt'
# Get class names and indexes
classnames,class_index = get_class_details(LABELMAP_PATH)

model_path = '/jidoka/simulations/jdisk2/jt.cust.Shriji_caps/sim_object_detection/sim_1/analysis/infer_model_7000'

# model_path = "/jidoka_training/simulations/jdisk2/Karma_XRay/M3_S1-1/2/faster_rcnn_inception_v2_coco/sim_6/analysis/infer_model_5000"

path_to_ckpt = os.path.join(model_path, 'frozen_inference_graph.pb')

# Image path
input_img_dir = '/jidoka/workspace/jdisk2/Shahabas/i/preprocessed/'


xml_save_dir = "/jidoka/workspace/jdisk2/Shahabas/i/xml"

make_sure_path_exists(xml_save_dir)
resize_image_shape = (512,512)     # Recommended to keep similar to train image resize shape


classwise_threshold = [None, 0.3,0.5,0.3,0.7,0.7,0.88,0.5,0.5,0.7,0.30,0.7,0.7,0.79,0.85,0.5,0.7,0.5,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7]

detection_threshold = 0.1

################################ FUNCTIONS ######################################################
def get_class_details(path_to_pbtxt):
    """ Function to get class details from label.pbtxt file.
        Parses labels.pbtxt file to get class names and their index for confusion matrix.
        This adds an additional class "OK" for the purpose of displaying confusion matrix. .

        KeyWord Arguments
        path_to_pbtxt : path to labels.pbtxt file (No default).
        
        Output 
        returns tuple of class and a dictionary containing its index of occurence (from 0).
    """    
    class_index = {}
    with open(path_to_pbtxt) as f:
        f_str = f.read()
        f_list = f_str.split('item')
        for item in f_list:
            if len(item) > 2:
                item_list = item.split('\n')
                class_index[item_list[2].split("'")[1]] = int(item_list[1].split(':')[1])

    
    class_names = [None]*len(class_index)
    
    index = []
    
    
    for c in class_index:
        index += [class_index[c]]
        
    class_names = [None]*(max(index)+1)

            
    print(max(index))  
        
    for c in class_index:
        
        class_index[c]
        class_names[class_index[c]] = c

    return class_names,class_index

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
def show_image(img):
    plt.figure(figsize=(16,16))
    plt.axis('off')
    if len(img.shape)==2 or (len(img.shape)==1 and img.shape[-1]==1):
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.show()
    
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def postprocess(detection_boxes, detection_classes, detection_scores):
    filtered_boxes, filtered_classes, filtered_scores = [], [], []
    
    for b, c, s in zip(detection_boxes, detection_classes, detection_scores):
        if s>=(classwise_threshold[c]):
            filtered_boxes += [b]
            filtered_classes += [c]
            filtered_scores += [s]
            
    filtered_boxes = np.asarray(filtered_boxes)
    filtered_classes = np.asarray(filtered_classes)
    filtered_scores = np.asarray(filtered_scores)
    return filtered_boxes, filtered_classes, filtered_scores

def save_to_xml(detection_classes,detection_boxes,xml_save_path,png_save_path,image_shape):
    png_filename = os.path.basename(png_save_path)
    with open(xml_save_path, "w+") as f:
        f.write("<annotation>")
        f.write("\n \t<folder>"+"xml"+"</folder>")
        f.write("\n \t<filename>"+png_filename+"</filename>")
        f.write("\n \t<source>")
        f.write("\n \t\t<database>Unknown</database>")
        f.write("\n \t\t<annotation>Unknown</annotation>")
        f.write("\n \t\t<image>Unknown</image>")        
        f.write("\n \t</source>")      
        f.write("\n \t<size>")
        f.write("\n \t\t<width>"+str(image_shape[0])+"</width>")
        f.write("\n \t\t<height>"+str(image_shape[1])+"</height>")
        f.write("\n \t\t<depth></depth>")
        f.write("\n \t</size>")
        f.write("\n \t<segmented>0</segmented>")

        for i in range(len(detection_classes)):
            detection_class = detection_classes[i]
            detection_box = detection_boxes[i]            
            f.write("\n \t<object>")
            f.write("\n \t\t<name>"+detection_class+"</name>")
            f.write("\n \t\t<occluded>0</occluded>")
            f.write("\n \t\t<bndbox>")
            f.write("\n \t\t\t<xmin>"+str(round(detection_box[1]*image_shape[0],2))+"</xmin>")
            f.write("\n \t\t\t<ymin>"+str(round(detection_box[0]*image_shape[1],2))+"</ymin>")
            f.write("\n \t\t\t<xmax>"+str(round(detection_box[3]*image_shape[0],2))+"</xmax>")
            f.write("\n \t\t\t<ymax>"+str(round(detection_box[2]*image_shape[1],2))+"</ymax>")
            f.write("\n \t\t</bndbox>")
            f.write("\n \t</object>")

        f.write("\n </annotation>")
        
#######################################################################################################

# Load graph and start session
print('Loading Graph from:',path_to_ckpt)
graph = load_graph(path_to_ckpt)
sess = tf.Session(graph=graph)

filenames = os.listdir(input_img_dir)

###################################################### MAIN FUNCTION #########################################

class_list = []
filename_list = []
    
for f in filenames:
    print(f)
    filename = f#f.split(".png")[0]+f.split(".png")[1]+".png"
    if('.png' in filename):
        
        image_path = os.path.join(input_img_dir, f)
        try:
            image = Image.open(image_path)
        except Exception as e:
            print(str(e))
            continue

        if not (image.mode == 'RGB'):
            image = image.convert('RGB')
        image_shape = image.size
        
        image_np = load_image_into_numpy_array(image,None,None)

        image_np = cv2.resize(image_np, resize_image_shape)

        output_dict = run_inference_for_single_image(image_np, graph, sess)
#         output_dict = NMS(output_dict,iou_threshold=0.9)
        detection_classes = output_dict['detection_classes']
        detection_boxes = output_dict['detection_boxes']
        detection_scores = output_dict['detection_scores']
        category_index = {}
        for i, classname in enumerate(classnames):
            category_index[i] = {'name': classname}
                
        thresholded_classnames = []
        thresholded_boxes = []

        for i in range(len(detection_scores)):
            curr_score = detection_scores[i]
            if curr_score>detection_threshold:
                thresholded_classnames += [classnames[detection_classes[i]]]
                thresholded_boxes += [detection_boxes[i]]


        xml_save_path = os.path.join(xml_save_dir, filename[:-4]+'.xml')
        print(xml_save_path)
        save_to_xml(thresholded_classnames, thresholded_boxes, xml_save_path, image_path, image_shape)

        
