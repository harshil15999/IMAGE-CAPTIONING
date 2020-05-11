# -*- coding: utf-8 -*-
"""
Created on Fri May  8 23:36:05 2020

@author: Vishal Shah
"""
#pre config
path='../Flickr8k_text/Flickr8k.token.txt'
path1='F:/PROJECTS/IMAGE CAPTIONING/Flickr8k_text/Flickr8k.token.txt'
#%%
from pickle import load
import pandas as pd
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import json



#%%
def load_doc(path):
    file=open(path,'r')
    text=file.readlines()
    file.close()
    return text

text=load_doc(path1)  
print(text[0])  

#%%
#create a dictionary with the id and the text surrounding it
def load_description(doc):
    mapping=dict()
    for line in doc:
        #split line by white line space
        x=line.split()
        if(len(line)<2):
            continue
        
        image_id,image_desc=x[0],x[1:]
        image_id=image_id.split('.')[0]
        image_desc=' '.join(image_desc)
        if image_id not in mapping:
            mapping[image_id]=list()
        #storing it in list
        mapping[image_id].append(image_desc)
    return mapping

mapping=load_description(text)
print(len(mapping))
#%%
#cleaning thru the text
def clean_text(dict_map):
    '''
    Always pass in the dictionary containing the id and the associated text
    
    
    '''
    text_dict=dict_map
    for i in text_dict.keys():
        for j in range(len(text_dict[i])):
            #
            sentence=text_dict[i][j]
            tokenizer = nltk.RegexpTokenizer(r"\w+")
            new_words = tokenizer.tokenize(sentence)
            new_words=[x.lower() for x in new_words]
            new_words = [word for word in new_words if word.isalpha()]
            text_dict[i][j]=' '.join(new_words)
    
    return text_dict

mapping=clean_text(mapping)
            
#%%

# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
    '''
    

    Parameters
    ----------
    descriptions : This builds a vocabulary
        ALWAYS GIVE IN A DICTIONARY TYPE OBJECT

    Returns
    -------
    all_desc : int
        length of the unique vocalbulary set
    '''
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc

# summarize vocabulary
vocabulary = to_vocabulary(descriptions=mapping)
print('Vocabulary Size: %d' % len(vocabulary))
#%%


def save_file(dict1):

    from json  import dump
    with open('text_mappings.json','w') as fp:
        dump(dict1,fp)
        
        return True
'''
print(save_file(dict1=mapping))

json = json.dumps(mapping)
f = open("dict.json","w")
f.write(json)
f.close() '''
import pickle 
with open('text.pkl', 'wb') as f:
    pickle.dump(mapping, f)
    f.close()
    print(f.tell)