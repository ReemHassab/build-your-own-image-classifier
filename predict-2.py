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

#initialize variables with default values
arch=''
#------------------------------------------------
parser=argparse.ArgumentParser()
parser.add_argument('--checkpoint',action='store',default='chcekpoint.pth')
parser.add_argument('--image_path',action='store', type=str,default='flowers/test/1/image_06764.jpg')
parser.add_argument('--top_k',action='store',type=int, default=3)
parser.add_argument('--category_names',action='store',type=str,default='cat_to_name.json')
parser.add_argument('--gpu',action='store_true',help='use gpu if available')
args = parser.parse_args()
#------------------------------------------------
#select parameters from command line
if args.checkpoint:
    checkpoint=args.checkpoint
    
if args.image_path:
    filepath= args.image_path
    
if args.top_k:
    top_k=args.top_k
    
if args.category_names:
    filepath=args.category_names
    
if args.gpu: 
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#------------------------------------------------
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
def load_model(checkpoint_path):
    
    checkpoint = torch.load(checkpoint_path)
    
    #check which model the user chose
    if checkpoint['arch']=='vgg19':
        model.models.vgg19(pretrained=True)
        in_features=25088
        for param in model.parameters():
            para.requires_grad=False
    elif checkpoint['arch']=='alexnet': 
        model.models.alexnet(pretrained=True)
        in_features=9216
        for param in model.parameters():
            para.requires_grad=False
    elif checkpoint['arch']=='densenet121': 
        model.models.densenet121(pretrained=True)
        in_features=1024
        for param in model.parameters():
            para.requires_grad=False
            
    else:
        print("sorry archeticture not recognized")
        
    model.class_to_idx = checkpoint['class_to_idx']        
    hidden_units=checkpoint['hidden_units']
    
    
    classifier= nn.Sequential(OrderedDict([
                                ('fc1',nn.Linear(in_features,hidden_units)),
                                 ('relu',nn.ReLU()),
                                  ('drop', nn.Dropout(0.2)),
                                  ('fc2',nn.Linear(1024,102)),
                                   ('output',nn.LogSoftmax(dim=1))
                                ]))
    model.classifier=classifier
    model.load_state_dict(checkpoint['state_dict'])
        
    return model

#------------------------------------------------
def process_img(image):
    '''
    '''
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
#------------------------------------------------
load_checkpoint(checkpoint)
print('the model being used for prediction is: ')
print(model)
labels= predict(filepath,model,topk)
print('-' * 10)
print(labels)
print('-' * 10)

