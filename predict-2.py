import torch
from torch import nn
from torch import optim
from torch.autograd import variable
from torchvision import datasets,transforms,models
from torchvision.datasets import ImageFolder
import argparse
import torch.nn.functional as F
from PIL import Image
from collections import OrderedDict
import numpy as np
import time
import os
import random
import seaborn as sns
import json
import matplotlib.pyplot as plt

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',action='store',default='/home/workspace/ImageClassifier ')
    parser.add_argument('--filepath',dest='filepath',default='flowers/test/1/image_06764.jpg')
    parser.add_argument('--top_k',dest=top_k, default=3)
    parser.add_argument('--category_names',dest='category_names',default='cat_to_name.json')
    parser.add_argument('--gpu',action='store',default='gpu')
    
def process_img(image):
    ''''''
    if image.width>image.height:
        image.thumbnail((10000000,256))
    else:
        image.thumbnail((256,10000000))
    
    l=(image.width-224)/2
    b=(image.height-224)/2
    r=(l+224)/2
    t=(b+224)/2
    
    image=image.crop(l,b,r,t)
    
    image=np.array(image)
    image=image/255
    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    image=((image-mean)/std)
    image=image.transpose(2,0,1)
    #convert to tensor and return 
    return torch.tensor(image)

def rebuild_model(filepath):
    
    checkpoint = torch.load('/home/workspace/ImageClassifier/checkpoint.pth', map_location=lambda storage, loc:storage)
    model=getattr(models, checkpoint['name'])(pretraind=True)
    model.classifier=checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx=checkpoint['class_to_idx']
    return model
   #invoke, send checkpoint path                       
rebuild_model('/home/workspace/ImageClassifier/checkpoint.pth')                       
print(model)

def predict(image_path, model, top_k=5):
  
    ''' Predict the class (or classes) of an image using a trained deep learning model.'''
    
    #no need for GPU here
    model.to('cpu')
    #set model to evaluation 
    model.eval();
    #convert the image from NumPy to torch 
    torch_img= torch.from_numpy(np.expand_dims(process_image(image_path),
                                              axis=0)).type(torch.FloatTensor).to('cpu')
    #find probabilities
    log_probs=model.forward(torch_img)
    
    #convert it to linear scale
    linear_probs= torch.exp(log_probs)
    
    #find top 5 results
    top_probs, top_labels= linear_probs.topk(topk)
    
    top_probs = np.array(top_probs.detach())[0]
    
    top_labels = np.array(top_labels.detach())[0]
    
    #convert to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers                     
def main():
   args= parse_args()
   gpu=args.gpu
   model= rebuild_model(args.checkpoint)
   cat_to_name=load_cat_names(args.category_names)
   img_path=args.filepath
   probs, classes=predict(img_path, model, int(args.top_k))
   labels=[cat_to_name[str(index)] for index in classes] 
   probability=probs
   print('File to be selected is: ' + image_path)
   print(labels)
   print(probability)
   n=0 
   while n < len(labels):
        print("{} with a probability of {}".format(labels[n], probability[n]))
        n += 1 # cycle through

if __name__ == "__main__":
    main()                      
                          
    