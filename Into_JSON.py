#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json


# In[3]:


mydict=dict()
json_file = open ("280_870.json", "r")
data = json.loads(json_file.read())


# In[15]:


print(data[0]["component_id"])


# In[16]:


for i in range(len(data)):
    mydict=dict()
    mydict["detection_boxes"]=data[i]['images'][1]["rejection_cause"]["algorithm"]["detection_boxes"]
    mydict["detection_classes"]=data[i]['images'][1]["rejection_cause"]["algorithm"]["detection_classes"]
    json_object = json.dumps(mydict, indent=4)
    with open(data[i]["component_id"]+".json", "w") as outfile:
        outfile.write(json_object)


# In[ ]:




