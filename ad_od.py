from pymongo import MongoClient
import os
import numpy as np
import shutil
import cv2
import matplotlib.pyplot as plt

client=MongoClient(host=["localhost:27017"])
db=client["jidoka"]
component_name="282_901"
component=db[component_name]
work_order="282_901_test1"
jidoka_path="/home/shahabas/shahabas/"

folder_name ="wago_analysis"
folder_path = os.path.join(jidoka_path, folder_name,component_name,work_order)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)



for component_data in component.find():
    batch_id = component_data["batch_id"]
    aql_data = component_data["aql_metadata"]

    if work_order in batch_id:
        for i in range(len(aql_data)):
            od_box = component_data["aql_metadata"]["unique_in_OD"]["detection_boxes"]
            ad_box = component_data["aql_metadata"]["unique_in_AD"]["detection_boxes"]
            od_classnames = component_data["aql_metadata"]["unique_in_OD"]["detection_classnames"]
            ad_classnames = component_data["aql_metadata"]["unique_in_AD"]["detection_classnames"]
            
            if ((od_classnames not in ad_classnames) and (ad_classnames not in od_classnames)):
                for i in range(len(component_data["images"])):
                    od = component_data['images'][i]["module"]
                    if od == "object_detection":
                        od_image = component_data["images"][i]['image_path'].split("/")[-1]
                        if "OD.png" in od_image:
                            od_image_path = os.path.join(jidoka_path, component_data["images"][i]['image_path'])
                            
                            img = cv2.imread(od_image_path)
                            a = cv2.resize(img, (900, 1024), interpolation=cv2.INTER_AREA)
                            for i in od_box:
                                x1, y1 = int(i[0] * 1024), int(i[1] * 900)
                                x2, y2 = int(i[2] * 1024), int(i[3] * 900)
                                cv2.rectangle(a,(y2,x2),(y1,x1),(0,0,255),thickness=2)
                            for i in ad_box:
                                x1,y1=int(i[0]*1024),int(i[1]*900)
                                x2,y2=int(i[2]*1024),int(i[3]*900)
                                cv2.rectangle(a,(y2,x2),(y1,x1),(0,255,0),thickness=2)
                                d=cv2.resize(a,(2048,1056))
                                
                 
                
                save_path = os.path.join(folder_path,od_image)
                cv2.imwrite(save_path, d)
                
#             plt.imshow(d)
#             plt.show()
#             break
            