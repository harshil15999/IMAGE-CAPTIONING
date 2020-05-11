# -*- coding: utf-8 -*-
"""
Created on Tue May  5 23:29:03 2020

@author: Vishal Shah
"""
#%%
print('1')
import os
from pathlib import Path
from os import listdir
import pandas as pd
import numpy as np
import sklearn.model_selection as m
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.xception import preprocess_input,Xception
from keras.models import Model

#%%
x=Xception()
x.summary()
#%%
def extract_features(directory):
    dir1=directory
    #extracting features
    model=Xception()
    #removing the last output layer
    model.layers.pop()
    #defining the Model which will take the input the size of the imahe and output the values in the last layer
    model=Model(inputs=model.inputs,output=model.layers[-1].output)
    
    print(model.summary())
    features=dict()
    file=os.listdir(dir1)
    for name in file:
        path=os.path.join(dir1,name)
        image=load_img(path=path,target_size=(299,299))
        #conv to np array
        image=img_to_array(image)
        image=image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
        #preprocessing the input image according to the Xception model
        image=preprocess_input(image)
        feature=model.predict(image)    
        #get  the image id
        image_id=name.split('.')[0]
        features[image_id]=feature
        print('>%s'%name)
    return features
        
        
#%%
from pathlib import Path
from keras.preprocessing.image import load_img,img_to_array

#features=extract_features(directory='F:/PROJECTS/IMAGE CAPTIONING/Flicker8k_Dataset')

#%%
from pickle import dump
import pickle
print('Extracted Features: %d' % len(features))
# save to file
dump(features, open('features.pkl', 'wb'))




