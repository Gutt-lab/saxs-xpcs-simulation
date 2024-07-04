#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:36:59 2024

@author: tosson
"""
import matplotlib.pyplot as plt
import io
import eel 
from random import randint 
import base64
import time

  
eel.init("webUI")   
  
# Exposing the random_python function to javascript 
@eel.expose     
def random_python(): 
    print("Random function running") 
    return randint(1,100) 

    
@eel.expose     
def generate_graph(): 
    plt.figure()
    plt.plot([1, 2, 3, 4], [10, 20, 25, 30])
    plt.title('Sample Plot')
   
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
   
    # Encode the bytes buffer to a base64 string
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64


@eel.expose     
def t(): 
    time.sleep(5)
    return True

# Start the index.html file 
eel.start("index.html")


# class _SampleParametersContainer(dict):
#     def __getitem__(self,key):
#         return dict.__getitem__(self,key)
    
#     def __setitem__(self, key, value):
#         dict.__setitem__(self,key,value)
        
#     def __delitem__(self, key):
#         print('f')
#         dict.__delitem__(self,key)
        
#     def __iter__(self):
#         return dict.__iter__(self)
    
#     def __len__(self):
#         return dict.__len__(self)
    
#     def __contains__(self, x):
#         return dict.__contains__(self,x)



# class SampleEnviroment(object):    
#     def __init__(self, sample_dict):
#         self.__bar = SampleParametersContainer(sample_dict)

#     def get_parameter(self, key):
#         return self.__bar.get(key)
    
#     def set_parameter(self, key, value):
#         self.__bar[key] = value

#     def remove_parameter(self, key):
#         self.__bar.__delitem__(key)


