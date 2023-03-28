
from pymongo import MongoClient
import shutil
import os
from pprint import pprint




pprint("started")
client=MongoClient(host=["localhost:27017"])
db=client["jidoka"]
component_name="769_171"
component=db[component_name]





lis=[]
clss_list=[]
old="OD_result_1"
new="raw"
jidoka_path="/jidoka/v3.1.4"
work_order="796-171-4332222-27-3-23-6:am"


for component_data in component.find():
    batch_id=component_data["batch_id"]
    len_images=len(component_data["images"])
    
    if work_order in batch_id:
        for i in range(len_images):
            od=component_data['images'][i]["module"]
            if od=="object_detection":
                
               
                
                
        
                img_decision = component_data["images"][i]['decision']
                
                image_path1 = component_data["images"][i]['image_path_result']
                
                raw_path_image=image_path1.replace(old,new)
                
                od_image_path1=component_data["images"][i]['image_path']
                # print(od_image_path1)
                
                
                if img_decision==1:
                    
                    
                    class_name=component_data['images'][i]['rejection_cause']['algorithm']['detection_classnames']

                    if len(class_name)>0:
                        
                        for j in class_name:
                            clss_list.append(j)

                



                    try:
                        folders=["Result","RAW","OD"]
                        for y in folders:
                            os.makedirs(os.path.join(jidoka_path,"analysis",component_name,work_order,"NG",y))
                                
                    except:
                        pass
                        
                    try:

                        
                        od_image_path=os.path.join(jidoka_path,od_image_path1)
                        # print(od_image_path)
                        raw_img_path=os.path.join(jidoka_path,raw_path_image)
                        image_path = os.path.join(jidoka_path,image_path1)
                        
                        save_path=os.path.join(jidoka_path,"analysis",component_name,work_order,"NG","Result",image_path1.split("/")[-1])
                
                        # save_raw_path=os.path.join(jidoka_path,"analysis",component_name,work_order,"RAW",raw_path_image.split("/")[-1])
                        # print(od_image_path1.split("/")[-1])

                        save_od=os.path.join(jidoka_path,"analysis",component_name,work_order,"NG","OD",od_image_path1.split("/")[-1])
                        
                        
                        shutil.copy(image_path,save_path)
                        # shutil.copy(raw_img_path,save_raw_path)
                        shutil.copy(od_image_path,save_od)
                        
                                                
                        
                    except Exception as e:
                        pprint(e)
                    




# #                 if img_decision==2:
                    
# #                     try:
# #                         class_name=component_data['images'][i]['rejection_cause']['algorithm']['detection_classnames']
                       
# #                         if len(class_name)>0:
                        
# #                             for j in class_name:
# #                                 clss_list.append(j)
# #                     except:
# #                         pass

# #                     try:
# #                         folders=["Result","RAW","OD"]
# #                         for y in folders:
# #                             os.makedirs(os.path.join(jidoka_path,"analysis",component_name,work_order,"May_be",y))
                                
# #                     except:
# #                         pass
                    
# #                     try:

                        
# #                         # od_image_path=os.path.join(jidoka_path,od_image_path1)
# #                         # raw_img_path=os.path.join(jidoka_path,raw_path_image)
# #                         image_path = os.path.join(jidoka_path,image_path1)
                        
# #                         save_path=os.path.join(jidoka_path,"analysis",component_name,work_order,"May_be","Result",image_path1.split("/")[-1])
                
# #                         # save_raw_path=os.path.join(jidoka_path,"analysis","specificity_error",component_name,work_order,"RAW",raw_path_image.split("/")[-1])
                        
# #                         # save_od=os.path.join(jidoka_path,"analysis","specificity_error",component_name,work_order,"OD",od_image_path1.split("/")[-1])
                        
                        
# #                         shutil.copy(image_path,save_path)
# #                         # shutil.copy(raw_img_path,save_raw_path)
# #                         # shutil.copy(od_image_path,save_od)
                        
                                                
                        
# #                     except Exception as e:
# #                         pprint(e)
                        

                    
                        
                    
                    
                    
                    
for x in clss_list:
    lis.append((clss_list.count(x),x))
with open (os.path.join(jidoka_path,"analysis",component_name,work_order,"defect.txt"),"w") as f:
    for z in list(set(lis)):
        f.writelines(f"{z[1]} - {z[0]}\n")


pprint("completed")









