import os,shutil
import numpy as np
import logging
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import base64
import requests
import json , errno
from abc import ABCMeta
from abc import abstractmethod
import collections
# Set headless-friendly backend.
import matplotlib; matplotlib.use('Agg')  # pylint: disable=multiple-statements
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import six

def make_sure_path_exists(path):
    """
        Create directory for output files
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def show_image(img):
    plt.figure(figsize=(8,8))
    plt.axis('off')
    if len(img.shape)==2 or (len(img.shape)==1 and img.shape[-1]==1):
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.show()
    
def get_concat_images(img1, img2, individual_shape=(1024,1024)):
    img1 = cv2.resize(img1, individual_shape)
    img2 = cv2.resize(img2, individual_shape)
    return np.concatenate([img1, img2], axis=1)

def show_images_same_channels(img1, img2):
    plt.figure(figsize=(16,8))
    plt.axis('off')
    plt.imshow(np.concatenate([img1, img2], axis=1), cmap='gray')
    plt.show()
    
def show_images(img1, img2, individual_shape=(2048,2048)):
    img1 = cv2.cvtColor(cv2.resize(img1, individual_shape), cv2.COLOR_RGB2GRAY)
    img2 = cv2.resize(img2, individual_shape)
    
    plt.figure(figsize=(16,8))
    plt.axis('off')

    plt.imshow(np.concatenate([img1, img2], axis=1), cmap='gray')
    plt.show()

def load_json_entries(json_dump_path):
    json_entries = []
    with open(json_dump_path) as data_file:    
        for line in data_file.readlines():
            if len(line)<=1:
                continue
            new_entry = json.loads(line)
            json_entries += [new_entry]
    return json_entries

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
    
def load_json(json_path,filename):
    """
        Function to load json from checkpoint path.  
    """
    with open(os.path.join(json_path,filename), 'r') as f:
        content = f.read()
        try:
            prediction_dict = json.loads(content)["predictions"][0]
        except:
            prediction_dict = json.loads(content)#["predictions"][0]
        return prediction_dict

def save_json(save_dir,output_dict,png_filename):
        output_str = json.dumps(output_dict, cls=NumpyEncoder)
        if '.' in png_filename:
            filename_without_ext = os.path.splitext(png_filename)[0]
        else:
            filename_without_ext = png_filename
        json_save_path = os.path.join(save_dir, filename_without_ext+'.json')
        
        with open(json_save_path, 'w') as f:
            f.write(output_str)     
        print("Saving_Json",png_filename)  

class Infer_Single_Json:
    """
        Class to get inference results from json 

        Args:
            json_path: path for the model 
            log_level: log_level for debugging. Default is WARNING. 
    """

    def __init__(self, json_path, log_level=logging.WARNING):

        logging.basicConfig(format='%(levelname)s:%(message)s', level=log_level)        


        assert type(json_path)==str, "json_path should be a string"
        assert os.path.exists(json_path), "json_path doesn't exist, please define correct path"

        self.json_path = json_path
        
        
    def run_inference_for_single_image(self,filename):
        """
            Function to load json from checkpoint path.  
        """
        print(filename)
        with open(os.path.join(self.json_path,filename), 'r') as f:
            content = f.read()
            try:
                prediction_dict = json.loads(content)["predictions"][0]
            except:
                prediction_dict = json.loads(content)#["predictions"][0]
            return prediction_dict

def load_image(image_path,resize_image_shape=None):
    """
        Loads image and returns a numpy array
    """
    try:
        image = Image.open(image_path)
    except Exception as e:
        logging.error(str(e))
        
    if not (image.mode == 'RGB'):
        image = image.convert('RGB')
    
    if resize_image_shape is not None:
        image = image.resize(resize_image_shape)

    (im_width, im_height) = image.size
    
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def save_image(image_path, image):
    """
        saves numpy array image. (wrapper for opencv imwrite)
    """   
    cv2.imwrite(image_path, image)

def force_symlink(src_link, traget_link):
    if os.path.exists(traget_link):
        os.unlink(traget_link)
    os.symlink(src_link, traget_link)
    return

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
                class_index[item_list[2].split("'")[1]] = int(item_list[1].split(':')[1])-1
    class_index["OK"] = len(class_index)
    class_names = [None]*len(class_index)
    for c in class_index:
        class_index[c]
        class_names[class_index[c]] = c
    return class_names, class_index

# --------------------------------------------------------------------------------------------------------------------------------
# Functions related to Object detection model inference 
# --------------------------------------------------------------------------------------------------------------------------------

import time
from copy import deepcopy

def infer_model(class_names, img_dir,
                step_number, batch_size, im_height, im_width, 
                path_to_ckpt = None, prediction_dir=None, 
                limit_evals = None, 
                filenames = None, original_image_size = None):

    save_extension = 'txt'
    image_extension = 'png'
    
    if path_to_ckpt is None:
        print("Please define path_to_ckpt")
    if prediction_dir is None:
        print("Please define prediction_dir")
        
    if original_image_size is None:
        original_image_size = im_height
        
    #image_multiply_factor = original_image_size/im_height
    
        
    # sess = tf.InteractiveSession(graph=graph)

    batch_img_list = []
    batch_img = None
    filename_batch = []
    filename_batch_list = []
    
    if filenames is None:
        file_open_mode = "w+"
        TEST_IMAGE_PATHS = [filename for filename in os.listdir(img_dir) if image_extension in filename]
    else:
        file_open_mode = "r+"
        TEST_IMAGE_PATHS = filenames
  

    print('Loading Images...')
    
    if limit_evals is not None:
        TEST_IMAGE_PATHS = TEST_IMAGE_PATHS[:limit_evals]
        print('Limiting evaluations to', limit_evals, 'images')
        
    image_multiply_factor = {}
    for image_path in TEST_IMAGE_PATHS:

        Image=cv2.imread(img_dir+image_path)
        
        curr_width, curr_height, channels = Image.shape
        
        image_multiply_factor[image_path[:-3]+save_extension] = (curr_width, curr_height)
        
        ## Check for number of channels into 3 channels
        image_np = cv2.resize(Image, (im_height, im_width))
        #image_np = load_image_into_numpy_array(image, None, None)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        if batch_img is None:
            batch_img = image_np_expanded
            filename_batch = [image_path[:-3]+save_extension]
        else:
            batch_img = np.append(batch_img, image_np_expanded, axis=0)
            filename_batch += [image_path[:-3]+save_extension]
        if batch_img.shape[0]==batch_size:
            batch_img_list+=[batch_img]
            batch_img = None

            filename_batch_list += [filename_batch]
            filename_batch = []

    # Append remaining images to the last batch
    if batch_img is not None:
        batch_img_list+=[batch_img]
        filename_batch_list += [filename_batch]
        
        
    print('Loading Graph from:',path_to_ckpt)
    graph = load_graph(path_to_ckpt)

    with tf.Session(graph=graph) as sess:
        x = graph.get_tensor_by_name('prefix/image_tensor:0')
        detection_boxes = graph.get_tensor_by_name('prefix/detection_boxes:0')
        detection_scores = graph.get_tensor_by_name('prefix/detection_scores:0')
        detection_classes = graph.get_tensor_by_name('prefix/detection_classes:0')
        num_detections = graph.get_tensor_by_name('prefix/num_detections:0')

        pred_time = []
        output = []
        print('Starting Inference...')

        if not os.path.exists(os.path.dirname(prediction_dir)):
            os.mkdir(os.path.dirname(prediction_dir))

        for batch_number in range(len(batch_img_list)):
            prefetched_input = deepcopy(batch_img_list[batch_number])
            t1 = time.time()
            curr_output = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], \
                          feed_dict={'prefix/image_tensor:0':prefetched_input})
            t2 = time.time()
            output += [curr_output]
            pred_time += [t2-t1]

    try:
        avg_time = sum(pred_time[1:])/(len(TEST_IMAGE_PATHS)-len(batch_img_list[0]))
    except:
        avg_time = pred_time[0]    # sess = sess.close()
    print('Inference Complete.')
    print('Average Inference Time:', round(avg_time*1000), 'milliseconds\n')

    
    # Save the predictions to txt files
    print('Now saving files...')
    
    if not os.path.isdir(prediction_dir):
        os.makedirs(prediction_dir)
    
    for batch_number in range(len(filename_batch_list)):
        detection_boxes, detection_scores, detection_classes, num_detections = output[batch_number]
        for i in range(detection_boxes.shape[0]):
            # The below code will append results to existing result.
            with open(prediction_dir+"/"+ filename_batch_list[batch_number][i], file_open_mode) as txtfile:
                curr_width, curr_height = image_multiply_factor[filename_batch_list[batch_number][i]]
                for j in range(detection_boxes[i].shape[0]):
                    class_name = class_names[int(detection_classes[i][j])-1]
                    ymin, xmin, ymax, xmax = tuple(detection_boxes[i][j].tolist())
                    # ymin, xmin, ymax, xmax = int(curr_height*ymin), int(curr_width*xmin), int(curr_height*ymax), int(curr_width*xmax)
                    content = class_name+ ' ' + str(detection_scores[i][j]) + ' ' + str(xmin) + " "+str(ymin)+" "+str(xmax)+" "+str(ymax)+"\n"
                    txtfile.write(str(content))
    print('Completed.')

def load_image_into_numpy_array(image, im_height, im_width):
    if im_height is None or im_height is None:
        (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph 
    
    
# --------------------------------------------------------------------------------------------------------------------------------
# Functions related to confusion matrix 
# --------------------------------------------------------------------------------------------------------------------------------


def resolve_multiple_bounding_boxes(prediction_dir,filenames=None,nms_threshold=0.1,save_wonms=False):
    """ 
        Applies technique similar to Non-Max Suppression technique, on predicted .txt files.
        Eliminates multiple bounding boxes from a same location. Only keeps the confidence, class
        coordinates of most-confident prediction.
        
        Threshold for non-max suppression is 10%.
        
        This is a multi-class NMS implementation. Here the bounding box is evaluated for each of the 
        classes present. The class with the higher score is retained for further processing. 
            
        KeyWord Arguments
        prediction_dir: directory, where predictions are stored in .txt format.

        Output
        None. Only modifies each txt file (overwrites) to fewer number of clean bounding boxes.

    """   
    
    # TODO: read the files from prediction_dir as in function get_confusion matrix. Remove files_confusion_matrix from function
    
    if filenames == None: 
         filenames = os.listdir(prediction_dir)
            
    if save_wonms == True:
        if not os.path.exists(os.path.join(prediction_dir,'wonms/')):
            print(os.path.join(prediction_dir,'wonms/'))
            os.mkdir(os.path.join(prediction_dir,'wonms/'))
        
    for filename in filenames:
        # Check if the file exists in wonms directory
        if '.txt' not in filename:
            if '.png' in filename:
                filename = filename.replace('.png','.txt')
            else:
                print('Filename not having .png or .txt : ',filename)
                continue
        
        # copy file for saving
        if save_wonms:
            shutil.copy(prediction_dir+filename,os.path.join(prediction_dir,'wonms/')+ filename)
        
        with open(prediction_dir+"/"+ filename,"r+") as txtfile:
            lines_pred = txtfile.read()

        clusters = []
        clusters_class = []
        for pred_specs in lines_pred.split('\n'):
            if len(pred_specs)<4:
                continue
            prediction_list = pred_specs.split(' ')
            box_coord = list(map(float, prediction_list[2:]))
            pred_class = prediction_list[0]
            pred_prob = float(prediction_list[1])

            cluster_found = False
            for clustered_boxes in clusters:
                for clustered_box in clustered_boxes:
                    if bb_intersection_over_union(box_coord, clustered_box[0])>nms_threshold:
                        clustered_boxes += [(box_coord, pred_class, pred_prob)]

                        cluster_found = True
                        break
                if cluster_found:
                    break

            if not cluster_found:
                clusters += [[(box_coord, pred_class, pred_prob)]]

        cluster_candidates = []
        for cluster in clusters:
            candidate = max(cluster, key=lambda x: x[2])
            classes = {}
            for prediction in cluster:
                if prediction[1] not in classes:
                    classes[prediction[1]] = prediction[2]
                else:
                    classes[prediction[1]] += prediction[2]
            # Choose the class with higher confidence
            # In-turn this will suppress a detected bounding box with a 
            # lower score for the same defect
            candidate_class = max(classes, key=lambda x:classes[x])
            cluster_candidates += [(candidate[0], candidate_class, candidate[2])]

        with open(prediction_dir+"/"+ filename,"w+") as txtfile:
            for i in range(len(cluster_candidates)):
                xmin, ymin, xmax, ymax = cluster_candidates[i][0][:]
                content = cluster_candidates[i][1]+ ' ' + str(cluster_candidates[i][2]) + ' ' + str(xmin) + " "+str(ymin)+" "+str(xmax)+" "+str(ymax)+"\n"
                txtfile.write(str(content))
    print('Done.')
    return



def bb_intersection_over_union(boxA, boxB):
    """ 
        Function to get intersection over union.
        
        KeyWord Arguments
        boxA : list of coordinates for a boxA (No default).
        boxB : list of coordinates for a boxB (No default).
        
        Output 
        return iou value for boxA and boxB
    """    
    ## determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])


    # compute the area of intersection rectangle
    interArea = max(0, (xB - xA)) * max(0, (yB - yA))


    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    if boxAArea*boxBArea==0:
        if boxAArea==0 and boxBArea==0:   
            return 1
        else:
            return 0

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def is_better_match(pred_index, truth_index, curr_best):
    if pred_index>curr_best:
        return True
    return False

def has_error(cf_matrix):
    # Returns true if confusion matrix has non-diagonal non-zero elements
    no_of_classes = len(cf_matrix)-1
    for i in range(no_of_classes):
        for j in range(no_of_classes+1):
            if i!=j and cf_matrix[i][j]!=0:
                return True
    return False


def get_confusion_matrix_archived(class_index, prediction_dir, truth_dir, class_name_wise_threshold, filenames=None, iou_threshold=0.1, prefer_higher_index=False):
    """    
        Function to get confusion metrix.

        KeyWord Arguments
            class_index : dict, corresponding to index of class by taking its name(string) (No default)
            prediction_dir : path to the folder containing text files for the predictions (No default)
            truth_dir : path to the folder containing text files for the ground truth (No default)
            class_name_wise_threshold: Threshold value. Predictions below this value will get ignored for analysis
        
        Output
            1. confusion_matrix: 2d matrix containing predictions for all the files
            2. error_filenames: list of filenames containing at least one error
            3. confusion_matrix_files: 2d matrix, where each cell of this matrix is a list of filenames which contributed to 
                that cell count. Length of list of each cell is equal to the number value in the confusion_matrix.
            4. files_confusion_matrix: list of tuple. Tuple contains 2 values:
                a. confusion matrix: contains the 2-d confusion matrix for a perticular file.
                b. filename: contains the filename in string format

    """     
    # Some Initializations
    confusion_matrix = [[0]*len(class_index)  for _ in range(len(class_index))]
    confusion_matrix_files = [[set([])]*len(class_index)  for _ in range(len(class_index))]
    files_confusion_matrix = []
    misclassification = 0
    detection_errors = 0
    error_filenames = []

    # Parameters and assumptions
    OK_index = -1    # assumes OK is last in class_index
    
    if filenames == None:
        # Fetch all the filenames from prediction directory
        filenames = os.listdir(prediction_dir)
    
    for f in filenames:
        if '.txt' not in f:
            if '.png' in f:
                f = f.replace('.png','.txt')
            else:
                print('Filename not having .png or .txt : ',f)
                continue
        confusion_matrix_component = [[0]*len(class_index)  for _ in range(len(class_index))]
        try:
            prediction_list = [line.rstrip('\n') for line in open(prediction_dir+f, 'r')]
        except:
            print('Couldn\'t read prediction filename: ',f)
            continue
        try:
            groundtruth_list = [line.rstrip('\n') for line in open(truth_dir+f, 'r')]     # Read Groundtruth, if exists
        except:
            print('Couldn\'t read groundtruth filename: ',f)
            continue                                             # If groundtruth is not available, skip this file.

        predictions_memo = {}
        prediction_matches_found = []
        for groundtruth_bb_specs in groundtruth_list:
            curr_best = -1            # Initialization

            # Store groundtruth specifications into different variables
            truth_specs_splitted = groundtruth_bb_specs.split(' ')
            truth_class = truth_specs_splitted[0]
            truth_box = list(map(float, truth_specs_splitted[1:]))

            if truth_class.lower()=='ok':
                continue

            truth_index = class_index[truth_class]   # index for confusion matrix

            claiming_prediction_boxes = []

            for prediction_bb_specs in prediction_list:
                pred_specs_splitted = prediction_bb_specs.split(' ')
                predicted_class = pred_specs_splitted[0]
                prediction_prob = float(pred_specs_splitted[1])
                predicted_box = list(map(float, pred_specs_splitted[2:]))

                if prediction_prob>=class_name_wise_threshold[predicted_class]:

                    predictions_memo[prediction_bb_specs] = {'predicted_class':predicted_class,
                                                             'prediction_prob': prediction_prob,
                                                             'predicted_box': predicted_box,
                                                             'truth_class': truth_class,
                                                             'truth_box': truth_box
                                                            }

                    iou = bb_intersection_over_union(predicted_box, truth_box)      # Find IOU
                    if iou > iou_threshold:  # Check if IOU is above the threshold
                        pred_centroid_x = (predicted_box[0]+predicted_box[2])/2.0
                        pred_centroid_y = (predicted_box[1]+predicted_box[3])/2.0
                        pred_centroid = [pred_centroid_x, pred_centroid_y]


                        claiming_prediction_boxes += [predicted_box]

                        prediction_matches_found += [prediction_bb_specs]
                        pred_index = class_index[predicted_class]
                        # TODO (mohan): Bug doesn't apply for all cases
                        # TODO (mohan): Need documentation
                        if prefer_higher_index or is_better_match(pred_index, truth_index, curr_best):
                            if curr_best!=-1:
                                confusion_matrix[truth_index][curr_best] -= 1   # Remove second best match for groundtruth
                                confusion_matrix_component[truth_index][curr_best] -= 1
                            curr_best = pred_index
                            confusion_matrix[truth_index][pred_index] += 1      # Add the best match for groundtruth
                            confusion_matrix_component[truth_index][pred_index] += 1

            if len(claiming_prediction_boxes)==0:
                detection_errors += 1
                confusion_matrix[truth_index][OK_index] += 1
                confusion_matrix_component[truth_index][OK_index] += 1

        # When Extra Bounding Boxes are Predicted (Human Error)
        for prediction_bb_specs in predictions_memo:
            predicted_class = predictions_memo[prediction_bb_specs]['predicted_class']
            pred_index = class_index[predicted_class]
            if prediction_bb_specs not in prediction_matches_found:
                predicted_box = predictions_memo[prediction_bb_specs]['predicted_box']

                pred_centroid_x = (predicted_box[0]+predicted_box[2])/2.0
                pred_centroid_y = (predicted_box[1]+predicted_box[3])/2.0

                detection_errors += 1
                confusion_matrix[OK_index][pred_index] += 1
                confusion_matrix_component[OK_index][pred_index] += 1

        files_confusion_matrix += [(confusion_matrix_component, f)]
        
        for p in range(len(class_index)):
            for q in range(len(class_index)):
                if confusion_matrix_component[p][q]!=0:
                    confusion_matrix_files[p][q] = confusion_matrix_files[p][q].union(set([f])) 

        if has_error(confusion_matrix_component):
            error_filenames += [f]

    misclassification = sum([sum(i[:-1]) for i in confusion_matrix[:-1]]) - sum([confusion_matrix[i][i] for i in range(len(class_index))])

    missed_detections = sum([row[-1] for row in confusion_matrix])
    print('Network Analysis: ')            
    total_predictions = sum([sum(i[:-1]) for i in confusion_matrix])
    total_groundtruths = sum([sum(i) for i in confusion_matrix[:-1]])
    print("Total predictions =", total_predictions)
    print("Total groundtruths =", total_groundtruths)

    print("Total Detection Errors (Human Error + Missed Detections) =", detection_errors)
    print("Total Missed Detection Error =", missed_detections)

    print("Total Classification Errors (for objects that were detected) =", misclassification)

    print('Component-wise Analysis: ')
    print("Total number of files with at least one error (except human error) =", len(error_filenames))

    return confusion_matrix, error_filenames, confusion_matrix_files, files_confusion_matrix

# --------------------------------------------------------------------------------------------------------------------------------
# Functions related to confusion matrix analysis
# --------------------------------------------------------------------------------------------------------------------------------

def get_success_rate(unacceptable, confusion_matrix, class_index):
    """ 
        Function to obtain the success rate.
            
        KeyWord Arguments
            unacceptable: tuple pairs of class pairs that are unacceptable. tuple first entry indicates groundtruth, right 
                part of tupe indicates predicted class.
            confusion_matrix : confusion metrix for the prediction (No default).
            class_index :  index of classes 

        Output
            Success Rate Metric. Success rate is defined as the 1 minus unacceptable error %. 
        
    """      
    unacceptable_indices = [tuple(class_index[name] for name in pair) for pair in unacceptable]
    total_unacceptable = 0
    for unacceptable_pair in unacceptable_indices:
        total_unacceptable += confusion_matrix[unacceptable_pair[0]][unacceptable_pair[1]]
    
    total_bounding_boxes = sum([sum(i) for i in confusion_matrix])
    
    if total_bounding_boxes!=0:
        faliure_rate = total_unacceptable/total_bounding_boxes
    elif total_unacceptable!=0:
        faliure_rate = 1
    else:
        faliure_rate = 0
    success_rate = 1-faliure_rate
    
    return success_rate



def get_classwise_success_rate(unacceptable, confusion_matrix, class_index):
    """ 
        Function to obtain the success rate.
            
        KeyWord Arguments
            unacceptable: tuple pairs of class pairs that are unacceptable. tuple first entry indicates groundtruth, right 
                part of tupe indicates predicted class.
            confusion_matrix : confusion metrix for the prediction (No default).
            class_index :  index of classes 

        Output
            Success Rate Metric. Success rate is defined as the 1 minus unacceptable error %.
            
            If model is sensitive, and/or model is detecting human error (where humans have forgotten to label)
            , then the formula = (total # detections - unacceptable detections)/(total # ground truths). 
        
    """      
    unacceptable_indices = [tuple(class_index[name] for name in pair) for pair in unacceptable]
    total_unacceptable = 0

    success_rate = {}

    for name in class_index:
        index_value = class_index[name]
        column_check = [confusion_matrix[index_value][j] for j in range(len(confusion_matrix)) if (index_value,j) in unacceptable_indices]
        
        #ignoring last row which accounts for AING error
        row_check = [confusion_matrix[j][index_value] for j in range(len(confusion_matrix)) if (index_value,j) in unacceptable_indices]
        class_cell_value = confusion_matrix[index_value][index_value]
        total_class_occurences = sum(row_check)+sum(column_check)+class_cell_value
        if total_class_occurences == 0:
            success_rate[name] = None
        else:
            total_acceptable_instanes = sum([confusion_matrix[col][index_value] for col in range(len(confusion_matrix)) if (col,index_value) not in unacceptable_indices])
            success_rate[name] = total_acceptable_instanes/total_class_occurences
    
    return success_rate

# --------------------------------------------------------------------------------------------------------------------------------
# Functions related to visualization of confusion matrix
# --------------------------------------------------------------------------------------------------------------------------------

#imports
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh
import seaborn as sn
import pandas as pd


def get_new_fig(fn, figsize=[9,9]):
    """ Init graphics """
    fig1 = plt.figure(fn, figsize)
    ax1 = fig1.gca()   #Get Current Axis
    ax1.cla() # clear existing plot
    return fig1, ax1

def configcell_text_and_colors(array_df, lin, col, oText, facecolors, posi, fz, fmt, show_null_values=0):
    """
      config cell text and colors
      and return text elements to add and to dell
      @TODO: use fmt
    """
    text_add = []; text_del = [];
    cell_val = array_df[lin][col]
    tot_all = array_df[-1][-1]
    per = (float(cell_val) / tot_all) * 100
    curr_column = array_df[:,col]
    ccl = len(curr_column)

    #last line  and/or last column
    if(col == (ccl - 1)) or (lin == (ccl - 1)):
        #tots and percents
        if(cell_val != 0):
            if(col == ccl - 1) and (lin == ccl - 1):
                tot_rig = 0
                for i in range(array_df.shape[0] - 1):
                    tot_rig += array_df[i][i]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif(col == ccl - 1):
                tot_rig = array_df[lin][lin]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif(lin == ccl - 1):
                tot_rig = array_df[col][col]
                per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok
        else:
            per_ok = per_err = 0

        per_ok_s = ['%.2f%%'%(per_ok), '100%'] [per_ok == 100]

        #text to DEL
        text_del.append(oText)

        #text to ADD
        font_prop = fm.FontProperties(weight='bold', size=fz)
        text_kwargs = dict(color='w', ha="center", va="center", gid='sum', fontproperties=font_prop)
        lis_txt = ['%d'%(cell_val), per_ok_s, '%.2f%%'%(per_err)]
        lis_kwa = [text_kwargs]
        dic = text_kwargs.copy(); dic['color'] = 'g'; lis_kwa.append(dic);
        dic = text_kwargs.copy(); dic['color'] = 'r'; lis_kwa.append(dic);
        lis_pos = [(oText._x, oText._y-0.3), (oText._x, oText._y), (oText._x, oText._y+0.3)]
        for i in range(len(lis_txt)):
            newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i])
            #print 'lin: %s, col: %s, newText: %s' %(lin, col, newText)
            text_add.append(newText)
        #print '\n'

        #set background color for sum cells (last line and last column)
        carr = [0.27, 0.30, 0.27, 1.0]
        if(col == ccl - 1) and (lin == ccl - 1):
            carr = [0.17, 0.20, 0.17, 1.0]
        facecolors[posi] = carr

    else:
        if(per > 0):
            txt = '%s\n%.2f%%' %(cell_val, per)
        else:
            if(show_null_values == 0):
                txt = ''
            elif(show_null_values == 1):
                txt = '0'
            else:
                txt = '0\n0.0%'
        oText.set_text(txt)

        #main diagonal
        if(col == lin):
            #set color of the textin the diagonal to white
            oText.set_color('w')
            # set background color in the diagonal to blue
            facecolors[posi] = [0.35, 0.8, 0.55, 1.0]
        
        elif (col==ccl-2):
            oText.set_color('w')
            # set last row color in the diagonal to green
            facecolors[posi] = [0.8, 0.2, 0.2, 1.0]
        elif (lin==ccl-2):
            oText.set_color('w')
            # set last row color in the diagonal to green
            facecolors[posi] = [0.35, 0.8, 1.0, 1.0]
        else:
            oText.set_color('r')

    return text_add, text_del

def insert_totals(df_cm):
    """ insert total column and line (the last ones) """
    sum_col = []
    for c in df_cm.columns:
        sum_col.append( df_cm[c].sum() )
    sum_lin = []
    for item_line in df_cm.iterrows():
        sum_lin.append( item_line[1].sum() )
    df_cm['sum_lin'] = sum_lin
    sum_col.append(np.sum(sum_lin))
    df_cm.loc['sum_col'] = sum_col

def pretty_plot_confusion_matrix(df_cm, annot=True, cmap="Oranges", fmt='.2f', fz=11,
      lw=0.5, cbar=False, figsize=[8,8], show_null_values=0, pred_val_axis='y', gt_name="Groundtruth", pred_name="Predicted"):
    """
      print conf matrix with default layout (like matlab)
      params:
        df_cm          dataframe (pandas) without totals
        annot          print text in each cell
        cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
        fz             fontsize
        lw             linewidth
        pred_val_axis  where to show the prediction values (x or y axis)
                        'col' or 'x': show predicted values in columns (x axis) instead lines
                        'lin' or 'y': show predicted values in lines   (y axis)
    """
    if(pred_val_axis in ('col', 'x')):
        xlbl = pred_name
        ylbl = gt_name
    else:
        xlbl = gt_name
        ylbl = pred_name
        df_cm = df_cm.T

    # create "Total" column
    insert_totals(df_cm)

    #this is for print allways in the same window
    fig, ax1 = get_new_fig('Conf matrix default', figsize)

    #thanks for seaborn
    ax = sn.heatmap(df_cm, annot=annot, annot_kws={"size": fz}, linewidths=lw, ax=ax1,
                    cbar=cbar, cmap=cmap, linecolor='w', fmt=fmt)

    #set ticklabels rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, fontsize = 10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 25, fontsize = 10)

    # Turn off all the ticks
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    #face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    #iter in text elements
    array_df = np.array( df_cm.to_records(index=False).tolist() )
    text_add = []; text_del = [];
    posi = -1 #from left to right, bottom to top.
    for t in ax.collections[0].axes.texts: #ax.texts:
        pos = np.array( t.get_position()) - [0.5,0.5]
        lin = int(pos[1]); col = int(pos[0]);
        posi += 1
        #print ('>>> pos: %s, posi: %s, val: %s, txt: %s' %(pos, posi, array_df[lin][col], t.get_text()))

        #set text
        txt_res = configcell_text_and_colors(array_df, lin, col, t, facecolors, posi, fz, fmt, show_null_values)

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    #remove the old ones
    for item in text_del:
        item.remove()
    #append the new ones
    for item in text_add:
        ax.text(item['x'], item['y'], item['text'], **item['kw'])

    #titles and legends
    ax.set_title('Confusion matrix')
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    plt.tight_layout()  #set layout slim
    plt.show()

def plot_confusion_matrix_from_data(y_test, predictions, columns=None, annot=True, cmap="Oranges",
      fmt='.2f', fz=11, lw=0.5, cbar=False, figsize=[8,8], show_null_values=0, pred_val_axis='lin'):
    """
        plot confusion matrix function with y_test (actual values) and predictions (predic),
        whitout a confusion matrix yet
    """
    from sklearn.metrics import confusion_matrix
    from pandas import DataFrame

    #data
    if(not columns):
        #labels axis integer:
        ##columns = range(1, len(np.unique(y_test))+1)
        #labels axis string:
        from string import ascii_uppercase
        columns = ['class %s' %(i) for i in list(ascii_uppercase)[0:len(np.unique(y_test))]]

    confm = confusion_matrix(y_test, predictions)
    cmap = 'Oranges';
    fz = 11;
    figsize=[9,9];
    show_null_values = 2
    df_cm = DataFrame(confm, index=columns, columns=columns)
    pretty_plot_confusion_matrix(df_cm, fz=fz, cmap=cmap, figsize=figsize, show_null_values=show_null_values, pred_val_axis=pred_val_axis, gt_name=gt_name, pred_name=pred_name)


def showCFmatrix(confusion_matrix, class_names, gt_name="Groundtruth", pred_name="Predicted"):
    """ 
        Function to show confusion matrix with greater details and
        saperate colours for special cells.
            
        KeyWord Arguments
        confusion_matrix : confusion metrix for the prediction (No default).
        class_names : list of class names (No default).

        Output
        display the confusion metrix in the form of heatmap.  

    """   
    df_cm = pd.DataFrame(confusion_matrix, index=[i for i in class_names],
                         columns=[i for i in class_names])
    pretty_plot_confusion_matrix(df_cm, pred_val_axis='x', gt_name=gt_name, pred_name=pred_name)
    

def showSimpleCFmatrix(confusion_matrix, class_names):
    """ 
        Function to show confusion matrix.
            
        KeyWord Arguments
        confusion_matrix : confusion metrix for the prediction (No default).
        class_names : list of class names (No default).

        Output
        display the confusion metrix in the form of heatmap.  

    """   
    df_cm = pd.DataFrame(confusion_matrix, index=[i for i in class_names],
                         columns=[i for i in class_names])
    plt.figure(figsize=(7, 5))
    sn.heatmap(df_cm, annot=True)
    plt.xlabel("Prediction")
    plt.ylabel("Groundtruth")
    
    
    
# --------------------------------------------------------------------------------------------------------------------------------
# Functions related to visualization of predictctions 
# --------------------------------------------------------------------------------------------------------------------------------

import matplotlib.patches as patches


def compareDetections(filenames, image_dir, class_to_color, txt_dir1, txt_dir2, class_name_wise_threshold, img_size, title1=None, title2=None):
    """ 
        Function to compare predictions with another prediction or groundtruth.
            
        KeyWord Arguments
            filenames : list of strings containing filenames
            image_dir : image directory path
            class_to_color : dictionary containing class as a key and color as a value for that class (No default).
            txt_dir1 : path to the folder containing text files for comparision for the predictions or groundtruth
            txt_dir2 : path to the folder containing text files for comparision for the predictions or groundtruth
            class_name_wise_threshold: Hides predictions from visualization, below these thresholds.
            img_size: get image size to ignore bounding boxes at edges   
            title1: title for detections on the left side (from txt_dir1)
            title2: title for detections on the right side (from txt_dir2)
        
        Output
            display detections side by side. If the directory contains predictions with confidence, it will show 
            percentage alongside boxes. If the directory contains groundtruth labels, it will show "GT" alongside
            boxes.

    """    
    
    file_extention = 'png'    # Dot is not required
    
    for col_index, txt_name in enumerate(filenames):
        image_filename = txt_name[:-3]+file_extention
        image_path = image_dir+image_filename

        im = np.array(Image.open(image_path), dtype=np.uint8)
        im_shape = im.shape
        im_width = im_shape[0]
        im_height = im_shape[1]
        fig, ax = plt.subplots(ncols=2, nrows=1)
        fig.set_size_inches((17, 17))
        fig.add_subplot()
        ax[0].imshow(im, cmap='gray')
        if title1 is None:
            ax[0].set_title("dir_1")
        else:
            ax[0].set_title(title1)
        ax[1].imshow(im, cmap='gray')
        if title2 is None:
            ax[1].set_title("dir_2")
        else:
            ax[1].set_title(title2)
        
        
        lines_dir1 = [line.rstrip('\n') for line in open(os.path.join(txt_dir1,txt_name), 'r')]
        
        for pred_specs in lines_dir1:
            specs_list = pred_specs.split(' ')
            is_GT = False
            if len(specs_list)>=6:
                box_specs_index = 2
                prediction_probability = specs_list[1]
            else:
                box_specs_index = 1
                prediction_probability = 1.0
                is_GT = True
                
            classname = specs_list[0]
            
            if float(prediction_probability)>=class_name_wise_threshold[classname]:
                box_coord = list(map(float, specs_list[box_specs_index:]))
                box_x = int(box_coord[0]*im_width)
                box_y = int(box_coord[1]*im_height)
                width = int(box_coord[2]*im_width)-box_x
                height = int(box_coord[3]*im_height)-box_y
                
                if is_GT:
                    prediction_percent_str = "GT"
                else:
                    prediction_percent_str = str(float(prediction_probability)*100)[:5]+ "%"

                rect = patches.Rectangle((box_x,box_y), width, height,linewidth=1,edgecolor=class_to_color[classname],facecolor='none')
                ax[0].add_patch(rect)
                ax[0].text(box_x, box_y, prediction_percent_str,
                           horizontalalignment='left',
                           verticalalignment='bottom',
                           color='white') 
                
                
        lines_dir2 = [line.rstrip('\n') for line in open(os.path.join(txt_dir2,txt_name), 'r')]
        
        for pred_specs in lines_dir2:
            specs_list = pred_specs.split(' ')
            is_GT = False
            if len(specs_list)>=6:
                box_specs_index = 2
                prediction_probability = specs_list[1]
            else:
                box_specs_index = 1
                prediction_probability = 1.0
                is_GT = True
                
            classname = specs_list[0]
            
            if float(prediction_probability)>=class_name_wise_threshold[classname]:
                box_coord = list(map(float, specs_list[box_specs_index:]))
                box_x = int(box_coord[0]*im_width)
                box_y = int(box_coord[1]*im_height)
                width = int(box_coord[2]*im_width)-box_x
                height = int(box_coord[3]*im_height)-box_y
                
                if is_GT:
                    prediction_percent_str = "GT"
                else:
                    prediction_percent_str = str(float(prediction_probability)*100)[:5]+ "%"

                rect = patches.Rectangle((box_x,box_y), width, height,linewidth=1,edgecolor=class_to_color[classname],facecolor='none')
                ax[1].add_patch(rect)
                ax[1].text(box_x, box_y, prediction_percent_str,
                           horizontalalignment='left',
                           verticalalignment='bottom',
                           color='white') 

        plt.show()
    
def displayPredictions_archived(filenames, image_dir, class_to_color, prediction_dir, truth_dir, class_name_wise_threshold, img_size):
    """ 
        Function to show confusion metrix.
            
        KeyWord Arguments
            filenames : list of strings containing filenames
            image_dir : image directory path
            class_to_color : dictionary containing class as a key and color as a value for that class (No default).
            prediction_dir : path to the folder containing text files for the predictions
            truth_dir : path to the folder containing text files for the ground truth
            class_name_wise_threshold: Hides predictions from visualization, below these thresholds.
            img_size: get image size to ignore bounding boxes at edges   
        
        Output
            display all the predictions and ground truth on the images.  

    """    
    
    file_extention = 'png'    # Dot is not required
    
    for col_index, txt_name in enumerate(filenames):
        image_filename = txt_name[:-3]+file_extention
        
        print(image_filename)
        
        image_path = image_dir+image_filename

        im = np.array(Image.open(image_path), dtype=np.uint8)
        im_shape = im.shape
        im_width = im_shape[1]
        im_height = im_shape[0]
        fig, ax = plt.subplots(ncols=2, nrows=1, dpi=160)
        fig.set_size_inches((20, 20))
        # Remove the comment if using old conatiner 
        # fig.add_subplot()

        ax[0].axis("off")
        ax[1].axis("off")

        ax[0].imshow(im, cmap='gray')
        ax[0].set_title("Prediction")
        ax[1].imshow(im, cmap='gray')
        ax[1].set_title("Ground Truth")
        
        plt.subplots_adjust(wspace=0, hspace=0)
        
        lines_pred = [line.rstrip('\n') for line in open(prediction_dir+txt_name, 'r')]
         
        for pred_specs in lines_pred:
            prediction_list = pred_specs.split(' ')
            
            classname = prediction_list[0]
            prediction_probability = prediction_list[1]
            if float(prediction_probability)>=class_name_wise_threshold[classname]:
                box_coord = list(map(float, prediction_list[2:]))
                box_x = int(box_coord[0]*im_width)
                box_y = int(box_coord[1]*im_height)
                width = int(box_coord[2]*im_width)-box_x
                height = int(box_coord[3]*im_height)-box_y
                
                pred_centroid_x = (int(box_coord[0]*im_width)+int(box_coord[2])*im_width)/2.0
                pred_centroid_y = (int(box_coord[1]*im_height)+int(box_coord[3]*im_height))/2.0
                
                prediction_percent_str = str(float(prediction_probability)*100)[:5]+ "%"
#                 if pred_centroid_y>0.000001*img_size and pred_centroid_y<0.999999*img_size and \
#                pred_centroid_x>0.000001*img_size and pred_centroid_x<0.999999*img_size:

                rect = patches.Rectangle((box_x,box_y), width, height,linewidth=1,edgecolor=class_to_color[classname],facecolor='none')
                ax[0].add_patch(rect)
                ax[0].text(box_x, box_y, prediction_percent_str,
                           horizontalalignment='left',
                           verticalalignment='bottom',
                           color='white') 
#                 else:
#                     rect = patches.Rectangle((box_x,box_y), width, height,linewidth=1,edgecolor='white',facecolor='none')
#                     ax[0].add_patch(rect)
#                     ax[0].text(box_x, box_y, prediction_percent_str+' (ignored)',
#                                horizontalalignment='left',
#                                verticalalignment='bottom',
#                                color='red') 

        truth_pred = [line.rstrip('\n') for line in open(truth_dir+txt_name, 'r')]
        for truth_specs in truth_pred:
            truth_list = truth_specs.split(' ')
            
            classname = truth_list[0]
            box_coord = list(map(float, truth_list[1:]))
            
            box_x = int(box_coord[0]*im_width)
            box_y = int(box_coord[1]*im_height)
            width = int(box_coord[2]*im_width)-box_x
            height = int(box_coord[3]*im_height)-box_y

            rect = patches.Rectangle((box_x,box_y), width, height,linewidth=1,edgecolor=class_to_color[classname],facecolor='none')
            ax[1].add_patch(rect)
  
        plt.show()

    
# --------------------------------------------------------------------------------------------------------------------------------
# Functions related to Ensembling
# --------------------------------------------------------------------------------------------------------------------------------
    
def get_filenames_with_confusion(prediction_dir, ignore_confidence_below, ignore_confidence_above, min_predictions_overlap, iou_threshold=0.1):
    """ 
        Function to get components (files) having confusion.
        
        Confusion is defined for a model, if - 
        1. Prediction confidence for some defect is between ignore_confidence_below (say 5%) and ignore_confidence_above (say 15%) 
        2. At least min_predictions_overlap (say 2) bounding boxes (satisfying condition 1) have their mutual IOU greater than 0.1.0
        - If above two conditions exist, for any defect, the complete file is considered as confusion for that model.
            
        KeyWord Arguments
        prediction_dir : directory, where the prediction txt files are present
        ignore_confidence_below=0.05 : ignore all the predictions below 5 % (No default)
        ignore_confidence_above=0.15: if prediction is above 15%, no confusion exist for that prediction
                                      (Confusion may still exist in same file, due to some other prediction.)
        min_predictions_overlap=2: Explained in criteria 2. Least number of predictions, which if overlaps, implies that
                                   network is fairly confident that a defect is present.
                                   The basic idea behind this is, if multiple predictions with low condidence indicates
                                   the presence of a defect, then it is highly probable that there is actually a defect
                                   and the network can be said to be overall confident for detecting that defect.

        Output
        returns a list of filenames, having at least one instance, where network is unsure/confused.

    """ 
    file_extention = 'png'    # Dot is not required
    
    filenames = []
    
    for filename in os.listdir(prediction_dir):
        if '.txt' not in filename:
            continue
    
        need_second_opinion = False
        with open(prediction_dir+"/"+ filename,"r+") as txtfile:
            lines_pred = txtfile.read()

        clusters = []
        clusters_class = []
        for pred_specs in lines_pred.split('\n'):
            if len(pred_specs)<4:
                continue
            prediction_list = pred_specs.split(' ')
            box_coord = list(map(float, prediction_list[2:]))
            pred_class = prediction_list[0]
            pred_prob = float(prediction_list[1])

            cluster_found = False
            for clustered_boxes in clusters:
                for clustered_box in clustered_boxes:
                    if bb_intersection_over_union(box_coord, clustered_box[0])>iou_threshold:
                        clustered_boxes += [(box_coord, pred_class, pred_prob)]

                        cluster_found = True
                        break
                if cluster_found:
                    break

            if not cluster_found:
                clusters += [[(box_coord, pred_class, pred_prob)]]

        cluster_candidates = []
        for cluster in clusters:
            candidate = max(cluster, key=lambda x: x[2])
            if candidate[2]<ignore_confidence_above and candidate[2]>ignore_confidence_below and len(cluster)<min_predictions_overlap:
                filenames += [filename[:-3]+file_extention]
                break
                
    return filenames




# --------------------------------------------------------------------------------------------------------------------------------
# Functions related to json files
# --------------------------------------------------------------------------------------------------------------------------------
def load_txt_prediction(file_path , class_index,is_groundtruth=False):
    prediction_list = [line.rstrip('\n') for line in open(file_path, 'r')]
    prediction_dict = {'detection_boxes': [], 'detection_classes':[], 'detection_scores':[]}
    for prediction_bb_specs in prediction_list:
        pred_specs_splitted = prediction_bb_specs.split(' ')
        if pred_specs_splitted[0].lower()=='ok':
            continue
            #prediction_dict['detection_classes'].append(int(class_index["OK"])+1)
        else:
            prediction_dict['detection_classes'].append(int(class_index[pred_specs_splitted[0]])+1)
        
        if(is_groundtruth==True):
            prediction_dict['detection_scores'].append(float(1.0))
            curr_box = list(map(float, pred_specs_splitted[1:]))
        else:
            prediction_dict['detection_scores'].append(float(pred_specs_splitted[1]))
            curr_box = list(map(float, pred_specs_splitted[2:]))
        
        curr_box = np.asarray([curr_box[0], curr_box[1], curr_box[3], curr_box[2]])
        prediction_dict['detection_boxes'].append(curr_box)
            
    

    #print(prediction_dict)
    return prediction_dict    
    

def get_confusion_matrix(class_index, prediction_dir, truth_dir, class_name_wise_threshold, filenames=None, iou_threshold=0.1, prefer_higher_index=False):
    """    
        Function to get confusion metrix.

        KeyWord Arguments
            class_index : dict, corresponding to index of class by taking its name(string) (No default)
            prediction_dir : path to the folder containing text files for the predictions (No default)
            truth_dir : path to the folder containing text files for the ground truth (No default)
            class_name_wise_threshold: Threshold value. Predictions below this value will get ignored for analysis
        
        Output
            1. confusion_matrix: 2d matrix containing predictions for all the files
            2. error_filenames: list of filenames containing at least one error
            3. confusion_matrix_files: 2d matrix, where each cell of this matrix is a list of filenames which contributed to 
                that cell count. Length of list of each cell is equal to the number value in the confusion_matrix.
            4. files_confusion_matrix: list of tuple. Tuple contains 2 values:
                a. confusion matrix: contains the 2-d confusion matrix for a perticular file.
                b. filename: contains the filename in string format

    """     
    # Some Initializations
    confusion_matrix = [[0]*len(class_index)  for _ in range(len(class_index))]
    confusion_matrix_files = [[set([])]*len(class_index)  for _ in range(len(class_index))]
    files_confusion_matrix = []
    misclassification = 0
    detection_errors = 0
    error_filenames = []
    
    
    classnames = {}
    for classname,class_id in class_index.items():
        classnames[class_id]=classname
    
    
    
    # Parameters and assumptions
    OK_index = -1    # assumes OK is last in class_index
    if filenames == None:
        # Fetch all the filenames from prediction directory
        filenames = os.listdir(prediction_dir)
    
    pred_infer= Infer_Single_Json(prediction_dir)
    gt_infer = Infer_Single_Json(truth_dir)
    for f in filenames:
        predictions_memo = {}
        prediction_matches_found = []
        confusion_matrix_component = [[0]*len(class_index)  for _ in range(len(class_index))]
        try:
            if ".json" in f:
                prediction_list = pred_infer.run_inference_for_single_image(f)   
                groundtruth_list = gt_infer.run_inference_for_single_image(f)     
            if ".txt" in f:
                prediction_list = load_txt_prediction(os.path.join(prediction_dir,f) , class_index)
                groundtruth_list = load_txt_prediction(os.path.join(truth_dir,f),class_index,groundtruth=True)
        except:
            print("Can not find file: {}".format(f))
            continue

        if len(groundtruth_list['detection_classes'])==0 and len(prediction_list['detection_classes'])==0:
            confusion_matrix[OK_index][OK_index] += 1
            confusion_matrix_files[OK_index][OK_index] = confusion_matrix_files[OK_index][OK_index].union(set([f])) 
            confusion_matrix_component[OK_index][OK_index] += 1
            #files_confusion_matrix += [(confusion_matrix_component, f)]

        found_maybe = False
        for bb_coords in prediction_list['detection_boxes']:
            if bb_coords==[0,0,0,0]:
                found_maybe=True
                break
        for bb_coords in groundtruth_list['detection_boxes']:
            if bb_coords==[0,0,0,0]:
                found_maybe=True
                break

        if found_maybe:
            continue

        found_GTOK = False
        if len(groundtruth_list['detection_classes'])==0:
            found_GTOK = True
            groundtruth_list['detection_classes'] = [OK_index]
            groundtruth_list['detection_boxes'] = [[0,0,0,0]]
            groundtruth_list['detection_scores'] = [-1]

        for gt_index in range(len(groundtruth_list['detection_classes'])):
            curr_best = -1            # Initialization

            # Store groundtruth specifications into different variables
            groundtruth_detection_classes = groundtruth_list['detection_classes'][gt_index]

            truth_box = groundtruth_list['detection_boxes'][gt_index]
            truth_index = int(groundtruth_detection_classes)   # index for confusion matrix
            truth_index -= 1      # Subtracted 1, because eval script assumes classes starting from 0 not 1

            if not found_GTOK:
                truth_class = classnames[truth_index]
                if truth_class.lower()=='ok':
                    found_GTOK = True
            else:
                truth_class = "ok"

            claiming_prediction_boxes = []

            #for prediction_bb_specs in prediction_list:   
            prediction_bb_specs = []
            for pt_index in range(len(prediction_list['detection_classes'])):
                predicted_class_int = int(prediction_list['detection_classes'][pt_index])
                predicted_class_int -= 1   # Subtracted 1, because eval script assumes classes starting from 0 not 1
                predicted_box = prediction_list['detection_boxes'][pt_index]
                prediction_prob = prediction_list['detection_scores'][pt_index]
                predicted_class = classnames[predicted_class_int]                    

                if prediction_prob>=class_name_wise_threshold[predicted_class]:
                    predictions_memo[pt_index] = {'predicted_class':predicted_class,
                                                             'prediction_prob': prediction_prob,
                                                             'predicted_box': predicted_box,
                                                             'truth_class': truth_class,
                                                             'truth_box': truth_box
                                                            }

                    if found_GTOK:
                        continue

                    iou = bb_intersection_over_union(predicted_box, truth_box)      # Find IOU
                    if iou >= iou_threshold:  # Check if IOU is above the threshold
                        claiming_prediction_boxes += [predicted_box]

                        prediction_matches_found += [pt_index]

                        pred_index = class_index[predicted_class]
                        # TODO (mohan): Bug doesn't apply for all cases
                        # TODO (mohan): Need documentation

                        if prefer_higher_index or is_better_match(pred_index, truth_index, curr_best):
                            if curr_best!=-1:
                                confusion_matrix[truth_index][curr_best] -= 1   # Remove second best match for groundtruth
                                confusion_matrix_component[truth_index][curr_best] -= 1
                            curr_best = int(pred_index)
                            confusion_matrix[truth_index][pred_index] += 1      # Add the best match for groundtruth
                            confusion_matrix_component[truth_index][pred_index] += 1

            if not found_GTOK and len(claiming_prediction_boxes)==0:
                detection_errors += 1
                confusion_matrix[truth_index][OK_index] += 1
                confusion_matrix_component[truth_index][OK_index] += 1

        # When Extra Bounding Boxes are Predicted (Human Error)
        for pt_index in predictions_memo:
            predicted_class = predictions_memo[pt_index]['predicted_class']
            pred_index = class_index[predicted_class]
            if pt_index not in prediction_matches_found:

                predicted_box = predictions_memo[pt_index]['predicted_box']
                detection_errors += 1
                confusion_matrix[OK_index][pred_index] += 1
                confusion_matrix_component[OK_index][pred_index] += 1

        files_confusion_matrix += [(confusion_matrix_component, f)]

        for p in range(len(class_index)):
            for q in range(len(class_index)):
                if confusion_matrix_component[p][q]!=0:
#                         print("p:{}, q:{} ".format(p,q))

                    confusion_matrix_files[p][q] = confusion_matrix_files[p][q].union(set([f])) 

        if has_error(confusion_matrix_component):
            error_filenames += [f]

    misclassification = sum([sum(i[:-1]) for i in confusion_matrix[:-1]]) - sum([confusion_matrix[i][i] for i in range(len(class_index)-1)])

    missed_detections = sum([row[-1] for row in confusion_matrix]) - confusion_matrix[OK_index][OK_index]
    print('Network Analysis: ')            
    total_predictions = sum([sum(i[:-1]) for i in confusion_matrix])
    total_groundtruths = sum([sum(i) for i in confusion_matrix[:-1]])
    print("Total predictions =", total_predictions)
    print("Total groundtruths =", total_groundtruths)

    print("Total Detection Errors (Human Error + Missed Detections) =", detection_errors)
    print("Total Missed Detection Error =", missed_detections)

    print("Total Classification Errors (for objects that were detected) =", misclassification)

    print('Component-wise Analysis: ')
    print("Total number of files with at least one error (except human error) =", len(error_filenames))

    return confusion_matrix, error_filenames, confusion_matrix_files, files_confusion_matrix

def displayPredictions(filenames, image_dir, class_to_color, prediction_dir, truth_dir, class_name_wise_threshold, img_size, class_index=None, gt_window_name="Ground Truth", pred_window_name="Prediction"):
    """ 
        Function to show confusion metrix.
            
        KeyWord Arguments
            filenames : list of strings containing filenames
            image_dir : image directory path
            class_to_color : dictionary containing class as a key and color as a value for that class (No default).
            prediction_dir : path to the folder containing text files for the predictions
            truth_dir : path to the folder containing text files for the ground truth
            class_name_wise_threshold: Hides predictions from visualization, below these thresholds.
            img_size: get image size to ignore bounding boxes at edges   
        
        Output
            display all the predictions and ground truth on the images.  

    """    
    
    file_extention = '.png'    # Dot is not required

    if class_index is None:
        class_index = {}
        for fake_class_id, classname in enumerate(class_to_color.keys()):
             class_index[classname] = fake_class_id

    classnames = {}
    for classname,class_id in class_index.items():
        classnames[class_id]=classname
    
    

    truth_infer = Infer_Single_Json(truth_dir)
    pred_infer = Infer_Single_Json(prediction_dir)
        
    for col_index, filename in enumerate(filenames):
        image_filename = os.path.splitext(filename)[0]+file_extention
        image_path = image_dir+image_filename

        im = np.array(Image.open(image_path), dtype=np.uint8)
        im_shape = im.shape
        im_width = im_shape[1]
        im_height = im_shape[0]
        fig, ax = plt.subplots(ncols=2, nrows=1, dpi=160)
        fig.set_size_inches((20, 20))
        #fig.add_subplot()

        ax[1].axis("off")
        ax[0].axis("off")

        ax[1].imshow(im, cmap='gray')
        ax[1].set_title(pred_window_name)
        ax[0].imshow(im, cmap='gray')
        ax[0].set_title(gt_window_name)
        
        plt.subplots_adjust(wspace=0, hspace=0)
        
        if ".json" in filename:
            lines_pred = pred_infer.run_inference_for_single_image(os.path.join(prediction_dir,filename)) 
            truth_pred = truth_infer.run_inference_for_single_image(os.path.join(truth_dir,filename))
        elif ".txt" in filename:
            lines_pred = load_txt_prediction(os.path.join(prediction_dir,filename) , class_index)
            truth_pred = load_txt_prediction(os.path.join(truth_dir,filename),class_index,groundtruth=True)


        for gt_index in range(len(truth_pred['detection_classes'])):
            index = int(truth_pred['detection_classes'][gt_index])-1
            prediction_probability = truth_pred['detection_scores'][gt_index]
            classname = classnames[index]
            
            if float(prediction_probability)>=class_name_wise_threshold[classname]:
                box_coord = truth_pred['detection_boxes'][gt_index]
                curr_class = truth_pred['detection_classes'][gt_index]
                box_x = int(box_coord[0]*im_width)
                box_y = int(box_coord[1]*im_height)
                width = int(box_coord[2]*im_width)-box_x
                height = int(box_coord[3]*im_height)-box_y
                
                pred_centroid_x = (int(box_coord[0]*im_width)+int(box_coord[2])*im_width)/2.0
                pred_centroid_y = (int(box_coord[1]*im_height)+int(box_coord[3]*im_height))/2.0
                prediction_percent_str = "{}: {}%".format(curr_class ,int(prediction_probability*100))

                rect = patches.Rectangle((box_y, box_x), height, width,linewidth=1,edgecolor=class_to_color[classname],facecolor='none')
                ax[0].add_patch(rect)
                ax[0].text(box_y, box_x, prediction_percent_str,
                           horizontalalignment='left',
                           verticalalignment='bottom',
                           color='white')


        for pt_index in range(len(lines_pred['detection_classes'])):
            index = int(lines_pred['detection_classes'][pt_index])-1
            prediction_probability = lines_pred['detection_scores'][pt_index]
            classname = classnames[index]
            
            if float(prediction_probability)>=class_name_wise_threshold[classname]:
                box_coord = lines_pred['detection_boxes'][pt_index]
                curr_class = lines_pred['detection_classes'][pt_index]
                box_x = int(box_coord[0]*im_width)
                box_y = int(box_coord[1]*im_height)
                width = int(box_coord[2]*im_width)-box_x
                height = int(box_coord[3]*im_height)-box_y
                
                pred_centroid_x = (int(box_coord[0]*im_width)+int(box_coord[2])*im_width)/2.0
                pred_centroid_y = (int(box_coord[1]*im_height)+int(box_coord[3]*im_height))/2.0
                
                prediction_percent_str = "{}: {}%".format(curr_class ,int(prediction_probability*100))

                rect = patches.Rectangle((box_y, box_x), height, width,linewidth=1,edgecolor=class_to_color[classname],facecolor='none')
                ax[1].add_patch(rect)
                ax[1].text(box_y, box_x, prediction_percent_str,
                           horizontalalignment='left',
                           verticalalignment='bottom',
                           color='white')
        plt.show()








# --------------------------------------------------------------------------------------------------------------------------------
# Functions related to merging defct-instance wise confusion matrix to image-wise confusion matrix
# --------------------------------------------------------------------------------------------------------------------------------


def merge_date_ranges(data, check_window_time=5):
    result = []
    sorted_times = sorted(data, key=lambda t: t[0])
    
    time_window = datetime.timedelta(seconds=int(check_window_time*30))
    
    c_group = []
    base_time = sorted_times[0][0]
        
    for ti in sorted_times:
        curr_time = ti[0]
        if (curr_time-base_time)<time_window:
            c_group.append(ti[2])
        else:
            result.append(c_group)
            c_group = [ti[2]]
        
            base_time = curr_time
            
    result.append(c_group)
   
    return result

def resolve_component_conflict(c_list):
    num_conflicts = len(c_list)
    times = []
    time_window = 1
    to_subtract_time = datetime.timedelta(seconds=(time_window*60))
    for i in range(num_conflicts):
        filename = c_list[i]
        datetime_splitted = filename.split("_")[0]
        date_str, time_str = datetime_splitted.split('-')

        yyd = int("20"+date_str[:2])
        mmd = int(date_str[2:4])
        ddd = int(date_str[4:])

        hht = int(time_str[:2])
        mmt = int(time_str[2:4])
        sst = int(time_str[4:])

        ti = datetime.datetime(yyd, mmd, ddd, hht, mmt, sst)

        times.append((ti, i, filename))

    merged_filenames = merge_date_ranges(times, time_window)
    return merged_filenames
    

def merge_components(filenames):
    c_maps = {}

    for filename in filenames:
        if len(filename)<25:
            continue
        c_id = filename.split("_C_")[1].split("_I_")[0]
        c_id = int(c_id)

        if c_id not in c_maps:
            c_maps[c_id] = []

        c_maps[c_id].append(filename)
    
    
    unique_cmaps = {}
    total_uniques = 0
    time_diff_list = []
    single_files = 0
    unique_component_list = []
    
    for c_id, c_list in c_maps.items():
        merged_filenames = resolve_component_conflict(c_list)    
        #print(merged_lists)
        unique_cmaps[c_id] = merged_filenames
        
        unique_component_list += merged_filenames
                
    return unique_component_list

def merge_cfm(cfm):
    cfm = np.asmatrix(cfm)
    n_rows, n_cols = cfm.shape

    merged_cfm = np.zeros_like(cfm)
    nonzero_row, nonzero_col = np.nonzero(cfm)
        
    if len(nonzero_row)==0:
        return cfm, (-1,-1)       
    
    for i in range(len(nonzero_row)):
        if nonzero_row[i]==nonzero_col[i]:
            merged_cfm[nonzero_row[i], nonzero_col[i]] = 1
            return merged_cfm, (nonzero_row[i], nonzero_col[i])
        
    for i in range(len(nonzero_row)):
        for j in range(len(nonzero_col)):
            if nonzero_row[i]==nonzero_col[j] and nonzero_row[i]!=n_rows-1:
                merged_cfm[nonzero_row[i], nonzero_col[j]] = 1
                return merged_cfm, (nonzero_row[i], nonzero_col[j])
        
    for i in range(len(nonzero_row)):
        if nonzero_row[i]!=n_rows-1 and nonzero_col[i]!=n_cols-1:
            merged_cfm[nonzero_row[i], nonzero_col[i]] = 1
            return merged_cfm, (nonzero_row[i], nonzero_col[i])
    for i in range(len(nonzero_row)):
        if nonzero_col[i]!=n_cols-1:
            merged_cfm[nonzero_row[i], nonzero_col[i]] = 1
            return merged_cfm, (nonzero_row[i], nonzero_col[i])
    for i in range(len(nonzero_row)):
        merged_cfm[nonzero_row[i], nonzero_col[i]] = 1
        return merged_cfm, (nonzero_row[i], nonzero_col[i])


def merge_preds_imagewise(confusion_matrix, files_confusion_matrix, confusion_matrix_files):
    files_confusion_matrix_map = {}

    for cfm_tuple in files_confusion_matrix:
        f_cfm = cfm_tuple[0]
        filename = cfm_tuple[1]
        files_confusion_matrix_map[filename] = f_cfm
            
    imagewise_cfm = np.zeros_like(confusion_matrix)
    n_classes = len(confusion_matrix)
    merged_confusion_matrix_files = [[set([])]*n_classes  for _ in range(n_classes)]
    #imagewise_cfm[-1, -1] = confusion_matrix[-1][-1]
    merged_confusion_matrix_files[-1][-1] = confusion_matrix_files[-1][-1]
    for filename, f_cfm in files_confusion_matrix_map.items():
        merged_cfm, coords = merge_cfm(f_cfm)
        if coords!=(-1,-1):
            p = coords[0]
            q = coords[1]
            merged_confusion_matrix_files[p][q] = merged_confusion_matrix_files[p][q].union(set([filename]))
        imagewise_cfm += merged_cfm
        
        
    return imagewise_cfm, files_confusion_matrix, merged_confusion_matrix_files




import io
def get_img_from_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def displayPredictions_test(filenames, image_dir, class_to_color, prediction_dir, truth_dir, class_name_wise_threshold, img_size, class_index=None, gt_window_name="Ground Truth", pred_window_name="Prediction",display_selected_cells=True):
    """ 
        Function to show confusion metrix.
            
        KeyWord Arguments
            filenames : list of strings containing filenames
            image_dir : image directory path
            class_to_color : dictionary containing class as a key and color as a value for that class (No default).
            prediction_dir : path to the folder containing text files for the predictions
            truth_dir : path to the folder containing text files for the ground truth
            class_name_wise_threshold: Hides predictions from visualization, below these thresholds.
            img_size: get image size to ignore bounding boxes at edges   
        
        Output
            display all the predictions and ground truth on the images.  

    """    
    
    file_extention = '.png'    # Dot is not required
    plot_img_np = []
    file = []
    if class_index is None:
        class_index = {}
        for fake_class_id, classname in enumerate(class_to_color.keys()):
             class_index[classname] = fake_class_id

    classnames = {}
    for classname,class_id in class_index.items():
        classnames[class_id]=classname
    
    

    truth_infer = Infer_Single_Json(truth_dir)
    pred_infer = Infer_Single_Json(prediction_dir)
        
    for col_index, filename in enumerate(filenames):
        image_filename = os.path.splitext(filename)[0]+file_extention
        image_path = image_dir+image_filename

        im = np.array(Image.open(image_path), dtype=np.uint8)
        im_shape = im.shape
        im_width = im_shape[1]
        im_height = im_shape[0]
        
        #if(get_images != True):
            
        fig, ax = plt.subplots(ncols=2, nrows=1, dpi=160)
        fig.set_size_inches((30, 15))
        fig.add_subplot()

        ax[1].axis("off")
        ax[0].axis("off")

        ax[1].imshow(im, cmap='gray')
        ax[1].set_title(pred_window_name)
        ax[0].imshow(im, cmap='gray')
        ax[0].set_title(gt_window_name)

        plt.subplots_adjust(wspace=0, hspace=0)
        
        if ".json" in filename:
            lines_pred = pred_infer.run_inference_for_single_image(os.path.join(prediction_dir,filename)) 
            truth_pred = truth_infer.run_inference_for_single_image(os.path.join(truth_dir,filename))
        elif ".txt" in filename:
            lines_pred = load_txt_prediction(os.path.join(prediction_dir,filename) , class_index)
            truth_pred = load_txt_prediction(os.path.join(truth_dir,filename),class_index,groundtruth=True)

        file += [filename]
        for gt_index in range(len(truth_pred['detection_classes'])):
            index = int(truth_pred['detection_classes'][gt_index])-1
            prediction_probability = truth_pred['detection_scores'][gt_index]
            classname = classnames[index]
            
            if float(prediction_probability)>=class_name_wise_threshold[classname]:
                box_coord = truth_pred['detection_boxes'][gt_index]
                curr_class = truth_pred['detection_classes'][gt_index]
                box_x = int(box_coord[0]*im_width)
                box_y = int(box_coord[1]*im_height)
                width = int(box_coord[2]*im_width)-box_x
                height = int(box_coord[3]*im_height)-box_y
                
                pred_centroid_x = (int(box_coord[0]*im_width)+int(box_coord[2])*im_width)/2.0
                pred_centroid_y = (int(box_coord[1]*im_height)+int(box_coord[3]*im_height))/2.0
                prediction_percent_str = "{}: {}%".format(curr_class ,int(prediction_probability*100))

                rect = patches.Rectangle((box_y, box_x), height, width,linewidth=1,edgecolor=class_to_color[classname],facecolor='none')
                ax[0].add_patch(rect)
                ax[0].text(box_y, box_x, prediction_percent_str,
                           horizontalalignment='left',
                           verticalalignment='bottom',
                           color='white')


        for pt_index in range(len(lines_pred['detection_classes'])):
            index = int(lines_pred['detection_classes'][pt_index])-1
            prediction_probability = lines_pred['detection_scores'][pt_index]
            classname = classnames[index]
            
            if float(prediction_probability)>=class_name_wise_threshold[classname]:
                box_coord = lines_pred['detection_boxes'][pt_index]
                curr_class = lines_pred['detection_classes'][pt_index]
                box_x = int(box_coord[0]*im_width)
                box_y = int(box_coord[1]*im_height)
                width = int(box_coord[2]*im_width)-box_x
                height = int(box_coord[3]*im_height)-box_y
                
                pred_centroid_x = (int(box_coord[0]*im_width)+int(box_coord[2])*im_width)/2.0
                pred_centroid_y = (int(box_coord[1]*im_height)+int(box_coord[3]*im_height))/2.0
                
                prediction_percent_str = "{}: {}%".format(curr_class ,int(prediction_probability*100))

                rect = patches.Rectangle((box_y, box_x), height, width,linewidth=1,edgecolor=class_to_color[classname],facecolor='none')
                ax[1].add_patch(rect)
                ax[1].text(box_y, box_x, prediction_percent_str,
                           horizontalalignment='left',
                           verticalalignment='bottom',
                           color='white')
                
        plot_img_np += [get_img_from_fig(fig)]
        
        plt.show()
    return plot_img_np,file

class Infer_From_Model:
    """
        Class to get inference results from tensorflow saved model 

        Args:
            model_path: path for the model 
            log_level: log_level for debugging. Default is WARNING. 
    """

    def __init__(self, model_path, gpu_id="0", log_level=logging.WARNING):

        self.gpu_id = gpu_id

        self.tf_config = tf.ConfigProto()
        self.tf_config.gpu_options.allow_growth=True


        # tf.config.experimental.set_visible_devices(gpus[int(gpu_id)], 'GPU')
        # for gpu in gpus:
        #     tf.config.experimental.set_memory_growth(gpu, True)
        # logging.basicConfig(format='%(levelname)s:%(message)s', level=log_level)        


        assert type(model_path)==str, "model_path should be a string"
        assert os.path.exists(model_path), "model_path doesn't exist, please define correct path"

        self.model_path = model_path
        self.path_to_ckpt = os.path.join(model_path, 'frozen_inference_graph.pb')
        
        self._load_graph()
        self._create_session()

    def _load_graph(self):
        """
            Function to load graph from checkpoint path.  
        """
        with tf.device('/gpu:{}'.format(self.gpu_id)):
            detection_graph = tf.Graph()
            with detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')

            logging.info("Loading Graph from: %s" % self.path_to_ckpt)

        self.graph = detection_graph

    def _create_session(self):
        """
            Function to create tensorflow session when
            Infer_From_Model class is created
        """

        logging.info("Creating tensorflow session")

        self.sess = tf.Session(graph=self.graph, config=self.tf_config)

    def run_inference_for_single_image_features(self, image):
        with self.graph.as_default():
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [ 'num_detections', 'detection_boxes', 'detection_scores',
			        'detection_classes', 'detection_masks', 'detection_features']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[1], image.shape[2])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = self.sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})


            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[ 'detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            output_dict['detection_features'] = output_dict['detection_features'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    def run_inference_for_single_image(self, image):
        with self.graph.as_default():
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = self.sess.run(tensor_dict,
                                    feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict


class Infer_From_Serving:
    """
        Class to get inference results from tensorflow saved model 

        Args:
            model_path: path for the model 
            log_level: log_level for debugging. Default is WARNING. 
    """            
    def __init__(self,model_name,model_version,serving_address="localhost",serving_port="8501",shape=(1024,1024),log_level=logging.WARNING):

        #logging.basicConfig(format='%(levelname)s:%(message)s', level=log_level)        

        self.model_name = model_name
        self.model_version = model_version
        self.serving_address = serving_address
        self.serving_port = serving_port
        if type(shape) is list:
            shape = tuple(shape)
        self.shape = shape
        
    def run_inference_for_single_image(self,img):
        
        im_resize = cv2.resize(img,self.shape)
        URL = "http://"+self.serving_address+":"+self.serving_port+"/v1/models/"+self.model_name+"/versions/"+self.model_version+":predict" 
        headers = {"content-type": "application/json"}
        is_success, im_buf_arr = cv2.imencode(".png", im_resize)
        img_png = im_buf_arr.tobytes()
        image_content = base64.b64encode(img_png).decode("utf-8")#convert to 64 encode
        body = {
            "signature_name": "predict_images",
            "instances": [
                        {"images":{"b64":image_content}}
                        ]
            }

        r = requests.post(URL, data=json.dumps(body), headers = headers)


        pred_value = r.json()["predictions"][0]

        
        pred_value['model_name'] = self.model_name
        pred_value["detection_classes"] = np.asarray(pred_value["detection_classes"], dtype=int)
        return pred_value


class Infer_From_Serving_Clf:
    """
        Class to get inference results from tensorflow saved model 

        Args:
            model_path: path for the model 
            log_level: log_level for debugging. Default is WARNING. 
    """            
    def __init__(self,model_name,model_version,serving_address="localhost",serving_port="8501",shape=(1024,1024),log_level=logging.WARNING):

        logging.basicConfig(format='%(levelname)s:%(message)s', level=log_level)        

        self.model_name = model_name
        self.model_version = model_version
        self.serving_address = serving_address
        self.serving_port = serving_port
        if type(shape) is list:
            shape = tuple(shape)
        self.shape = shape
        
    def run_inference_for_single_image(self,img):
        
        im_resize = cv2.resize(img,self.shape)
        URL = "http://"+self.serving_address+":"+self.serving_port+"/v1/models/"+self.model_name+"/versions/"+self.model_version+":predict" 
        headers = {"content-type": "application/json"}
        is_success, im_buf_arr = cv2.imencode(".png", im_resize)
        img_png = im_buf_arr.tobytes()
        image_content = base64.b64encode(img_png).decode("utf-8")#convert to 64 encode
        body = {
            "signature_name": "predict",
            "instances": [
                        {"input_img":{"b64":image_content}}
                        ]
            }

        r = requests.post(URL, data=json.dumps(body), headers = headers)


        pred_list = r.json()["predictions"][0]
        
        boxes = []
        classes = []
        scores = []
        num_preds = 0
        default_box = [0,0,1,1]
        
        for i, class_pred_score in enumerate(pred_list):
            curr_class = i+1
            boxes.append(default_box)
            classes.append(curr_class)
            scores.append(class_pred_score)
            num_preds += 1      
        
        pred_value = {"detection_boxes": boxes, "detection_classes": classes, "detection_scores": scores, "num_detections": num_preds}
        pred_value['model_name'] = self.model_name

        return pred_value

from abc import ABCMeta
from abc import abstractmethod
import collections
# Set headless-friendly backend.
import matplotlib; matplotlib.use('Agg')  # pylint: disable=multiple-statements
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import six


_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]




def encode_image_array_as_png_str(image):
  """Encodes a numpy array into a PNG string.

  Args:
    image: a numpy array with shape [height, width, 3].

  Returns:
    PNG encoded image string.
  """
  image_pil = Image.fromarray(np.uint8(image))
  output = six.BytesIO()
  image_pil.save(output, format='PNG')
  png_string = output.getvalue()
  output.close()
  return png_string


def draw_bounding_box_on_image_array(image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color='red',
                                     thickness=4,
                                     display_str_list=(),
                                     use_normalized_coordinates=True):
  """Adds a bounding box to an image (numpy array).

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Args:
    image: a numpy array with shape [height, width, 3].
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                             thickness, display_str_list,
                             use_normalized_coordinates)
  np.copyto(image, np.array(image_pil))


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
  """Adds a bounding box to an image.

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
  try:
    font = ImageFont.truetype('/jidoka/code/jt.prod.training/visualization/arial.ttf', 30)
  except IOError:
    font = ImageFont.load_default()

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)
    text_bottom -= text_height - 2 * margin


def draw_bounding_boxes_on_image_array(image,
                                       boxes,
                                       color='red',
                                       thickness=4,
                                       display_str_list_list=()):
  """Draws bounding boxes on image (numpy array).

  Args:
    image: a numpy array object.
    boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
           The coordinates are in normalized format between [0, 1].
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list_list: list of list of strings.
                           a list of strings for each bounding box.
                           The reason to pass a list of strings for a
                           bounding box is that it might contain
                           multiple labels.

  Raises:
    ValueError: if boxes is not a [N, 4] array
  """
  image_pil = Image.fromarray(image)
  draw_bounding_boxes_on_image(image_pil, boxes, color, thickness,
                               display_str_list_list)
  np.copyto(image, np.array(image_pil))


def draw_bounding_boxes_on_image(image,
                                 boxes,
                                 color='red',
                                 thickness=4,
                                 display_str_list_list=()):
  """Draws bounding boxes on image.

  Args:
    image: a PIL.Image object.
    boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
           The coordinates are in normalized format between [0, 1].
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list_list: list of list of strings.
                           a list of strings for each bounding box.
                           The reason to pass a list of strings for a
                           bounding box is that it might contain
                           multiple labels.

  Raises:
    ValueError: if boxes is not a [N, 4] array
  """
  boxes_shape = boxes.shape
  if not boxes_shape:
    return
  if len(boxes_shape) != 2 or boxes_shape[1] != 4:
    raise ValueError('Input must be of size [N, 4]')
  for i in range(boxes_shape[0]):
    display_str_list = ()
    if display_str_list_list:
      display_str_list = display_str_list_list[i]
    draw_bounding_box_on_image(image, boxes[i, 0], boxes[i, 1], boxes[i, 2],
                               boxes[i, 3], color, thickness, display_str_list)


def _visualize_boxes(image, boxes, classes, scores, category_index, **kwargs):
  return visualize_boxes_and_labels_on_image_array(
      image, boxes, classes, scores, category_index=category_index, **kwargs)


def _visualize_boxes_and_masks(image, boxes, classes, scores, masks,
                               category_index, **kwargs):
  return visualize_boxes_and_labels_on_image_array(
      image,
      boxes,
      classes,
      scores,
      category_index=category_index,
      instance_masks=masks,
      **kwargs)


def _visualize_boxes_and_keypoints(image, boxes, classes, scores, keypoints,
                                   category_index, **kwargs):
  return visualize_boxes_and_labels_on_image_array(
      image,
      boxes,
      classes,
      scores,
      category_index=category_index,
      keypoints=keypoints,
      **kwargs)


def _visualize_boxes_and_masks_and_keypoints(
    image, boxes, classes, scores, masks, keypoints, category_index, **kwargs):
  return visualize_boxes_and_labels_on_image_array(
      image,
      boxes,
      classes,
      scores,
      category_index=category_index,
      instance_masks=masks,
      keypoints=keypoints,
      **kwargs)

def draw_keypoints_on_image_array(image,
                                  keypoints,
                                  color='red',
                                  radius=2,
                                  use_normalized_coordinates=True):
  """Draws keypoints on an image (numpy array).

  Args:
    image: a numpy array with shape [height, width, 3].
    keypoints: a numpy array with shape [num_keypoints, 2].
    color: color to draw the keypoints with. Default is red.
    radius: keypoint radius. Default value is 2.
    use_normalized_coordinates: if True (default), treat keypoint values as
      relative to the image.  Otherwise treat them as absolute.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_keypoints_on_image(image_pil, keypoints, color, radius,
                          use_normalized_coordinates)
  np.copyto(image, np.array(image_pil))


def draw_keypoints_on_image(image,
                            keypoints,
                            color='red',
                            radius=2,
                            use_normalized_coordinates=True):
  """Draws keypoints on an image.

  Args:
    image: a PIL.Image object.
    keypoints: a numpy array with shape [num_keypoints, 2].
    color: color to draw the keypoints with. Default is red.
    radius: keypoint radius. Default value is 2.
    use_normalized_coordinates: if True (default), treat keypoint values as
      relative to the image.  Otherwise treat them as absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  keypoints_x = [k[1] for k in keypoints]
  keypoints_y = [k[0] for k in keypoints]
  if use_normalized_coordinates:
    keypoints_x = tuple([im_width * x for x in keypoints_x])
    keypoints_y = tuple([im_height * y for y in keypoints_y])
  for keypoint_x, keypoint_y in zip(keypoints_x, keypoints_y):
    draw.ellipse([(keypoint_x - radius, keypoint_y - radius),
                  (keypoint_x + radius, keypoint_y + radius)],
                 outline=color, fill=color)


def draw_mask_on_image_array(image, mask, color='red', alpha=0.4):
  """Draws mask on an image.

  Args:
    image: uint8 numpy array with shape (img_height, img_height, 3)
    mask: a uint8 numpy array of shape (img_height, img_height) with
      values between either 0 or 1.
    color: color to draw the keypoints with. Default is red.
    alpha: transparency value between 0 and 1. (default: 0.4)

  Raises:
    ValueError: On incorrect data type for image or masks.
  """
  if image.dtype != np.uint8:
    raise ValueError('`image` not of type np.uint8')
  if mask.dtype != np.uint8:
    raise ValueError('`mask` not of type np.uint8')
  if np.any(np.logical_and(mask != 1, mask != 0)):
    raise ValueError('`mask` elements should be in [0, 1]')
  if image.shape[:2] != mask.shape:
    raise ValueError('The image has spatial dimensions %s but the mask has '
                     'dimensions %s' % (image.shape[:2], mask.shape))
  rgb = ImageColor.getrgb(color)
  pil_image = Image.fromarray(image)

  solid_color = np.expand_dims(
      np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
  pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
  pil_mask = Image.fromarray(np.uint8(255.0*alpha*mask)).convert('L')
  pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
  np.copyto(image, np.array(pil_image.convert('RGB')))


def visualize_boxes_and_labels_on_image_array(
    image,
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    agnostic_mode=False,
    line_thickness=4,
    groundtruth_box_visualization_color='black',
    skip_scores=False,
    skip_labels=False):
  """Overlay labeled boxes on an image with formatted scores and label names.

  This function groups boxes that correspond to the same location
  and creates a display string for each detection and overlays these
  on the image. Note that this function modifies the image in place, and returns
  that same image.

  Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    instance_masks: a numpy array of shape [N, image_height, image_width] with
      values ranging between 0 and 1, can be None.
    instance_boundaries: a numpy array of shape [N, image_height, image_width]
      with values ranging between 0 and 1, can be None.
    keypoints: a numpy array of shape [N, num_keypoints, 2], can
      be None
    use_normalized_coordinates: whether boxes is to be interpreted as
      normalized coordinates or not.
    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
      all boxes.
    min_score_thresh: minimum score threshold for a box to be visualized
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.  This mode will display scores but ignore
      classes.
    line_thickness: integer (default: 4) controlling line width of the boxes.
    groundtruth_box_visualization_color: box color for visualizing groundtruth
      boxes
    skip_scores: whether to skip score when drawing a single detection
    skip_labels: whether to skip label when drawing a single detection

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
  """
  # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.
  box_to_display_str_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)
  box_to_instance_masks_map = {}
  box_to_instance_boundaries_map = {}
  box_to_keypoints_map = collections.defaultdict(list)
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
    if scores is None or scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      if instance_masks is not None:
        box_to_instance_masks_map[box] = instance_masks[i]
      if instance_boundaries is not None:
        box_to_instance_boundaries_map[box] = instance_boundaries[i]
      if keypoints is not None:
        box_to_keypoints_map[box].extend(keypoints[i])
      if scores is None:
        box_to_color_map[box] = groundtruth_box_visualization_color
      else:
        display_str = ''
        if not skip_labels:
          if not agnostic_mode:
            if classes[i] in category_index.keys():
              class_name = category_index[classes[i]]['name']
            else:
              class_name = 'N/A'
            display_str = str(class_name)
        if not skip_scores:
          if not display_str:
            display_str = '{}%'.format(int(100*scores[i]))
          else:
            display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
        box_to_display_str_map[box].append(display_str)
        if agnostic_mode:
          box_to_color_map[box] = 'DarkOrange'
        else:
          box_to_color_map[box] = STANDARD_COLORS[
              classes[i] % len(STANDARD_COLORS)]

  # Draw all boxes onto image.
  for box, color in box_to_color_map.items():
    ymin, xmin, ymax, xmax = box
    if instance_masks is not None:
      draw_mask_on_image_array(
          image,
          box_to_instance_masks_map[box],
          color=color
      )
    if instance_boundaries is not None:
      draw_mask_on_image_array(
          image,
          box_to_instance_boundaries_map[box],
          color='red',
          alpha=1.0
      )
    draw_bounding_box_on_image_array(
        image,
        ymin,
        xmin,
        ymax,
        xmax,
        color=color,
        thickness=line_thickness,
        display_str_list=box_to_display_str_map[box],
        use_normalized_coordinates=use_normalized_coordinates)
    if keypoints is not None:
      draw_keypoints_on_image_array(
          image,
          box_to_keypoints_map[box],
          color=color,
          radius=line_thickness / 2,
          use_normalized_coordinates=use_normalized_coordinates)

  return image


def visualize_output_dict(img, output_dict, classnames=None, category_index=None, min_score_thresh = 0.0, max_boxes_to_draw = 100, line_thickness=2):
    boxes = np.asarray(output_dict["detection_boxes"])
    classes = np.asarray(output_dict["detection_classes"])
    scores = np.asarray(output_dict["detection_scores"])
        
    if category_index is None:
        if classnames is not None:
            if not (classnames[0] is None or classnames[0]=="None"):
                classnames = [None]+classnames
        else:
            classnames = ["class_{}".format(i) for i in range(100)]

        category_index = {}
        for i, classname in enumerate(classnames):
            category_index[i] = {'name': classname}    
    
    visualized_img = np.copy(img)       # To avoid in-place modifications
    visualized_img = visualize_boxes_and_labels_on_image_array(visualized_img, boxes, classes, scores, category_index, min_score_thresh=min_score_thresh, max_boxes_to_draw=max_boxes_to_draw,
                                    use_normalized_coordinates=True,
                                    line_thickness=line_thickness)

    return visualized_img
    

def visualize_output_dict_clf(img, output_dict, classnames=None, category_index=None, min_score_thresh = 0.0, max_boxes_to_draw = 100, line_thickness=2):
    boxes = np.asarray(output_dict["detection_boxes"])
    classes = np.asarray(output_dict["detection_classes"])
    scores = np.asarray(output_dict["detection_scores"])
    num_detections = len(classes)
            
    if category_index is None:
        if classnames is not None:
            if not (classnames[0] is None or classnames[0]=="None"):
                classnames = [None]+classnames
        else:
            classnames = ["class_{}".format(i) for i in range(100)]

        category_index = {}
        for i, classname in enumerate(classnames):
            category_index[i] = {'name': classname}    
    
    vis_image = np.copy(img)       # To avoid in-place modifications
    y0, dy = 10, 10
    
    for i in range(num_detections):
        pred_class_id = classes[i]
        predicted_classname = classnames[pred_class_id]
        prediction_score = "{}%".format(int(scores[i] * 100))
        
        y = y0 + i*dy
        prediction_str = "{}: {}".format(predicted_classname, prediction_score)
        
        cv2.putText(vis_image, prediction_str, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0))

    return vis_image
