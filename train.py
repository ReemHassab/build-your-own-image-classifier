#imports here
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
#---------------------------------------------------------
from workspace_utils import active_session
torch.cuda.is_available()

#---------------------------------------------------------
def parse_args():
    print("arguments being parsed")
    parser= argparse.ArgumentParser(description="Program            Training")
    parser.add_argument('--data_dir', action='store', default='./flowers/')
    parser.add_argument('--arch', dest='arch', default='vgg19')
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.002')
    parser.add_argument('--hidden_units', dest='hidden_units',default=512)
    parser.add_argument('--epochs',dest='epochs',default='8')
    parser.add_argument('--gpu', action='store',default='gpu')
    parser.add_argument('--save_dir',dest='save_dir',action='store',default="checkpoint.pth")
    return parser.parse_args()
#---------------------------------------------------------
#define a function for training
def training(model,epochs_number,criterion,optimizer,training_loader,validation_loader,current_device):
    
    device=current_device
    model.cuda()

    #trackings
    print_every=5
    running_loss=0
    accuracy=0
    steps=0
    run_accuracy=0
    train_losses, valid_losses = [], []
    train_accuracy, valid_accuracy = [], []



    #start training 
    start=time.time()
    print('begin training')

     # 1- training loop: looping through epochs
    model.train()
    for epoch in range (epochs_number):
        print('Epoch {}/{}'.format(epoch+1, epochs_number))
        print('-' * 10)
    
        # 2- looping through data
        for inputs,labels in training_loader:
            steps+=1
            
            #move input & label tensors to default device
            inputs, labels= inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
            #Forward
            log_ps=model.forward(inputs)
            loss=criterion(log_ps, labels)
            #Backward
            loss.backward()
            #take a step
            optimizer.step()
            #increment running loss, it's where we keep track of our running loss
            running_loss+=loss.item()
        
            # calculate the accuracy
            ps = torch.exp(log_ps) # get the actual probability
            top_p, top_class = ps.topk(1, dim=1) # top probabilities and classes
            equals = top_class == labels.view(*top_class.shape)
   
            run_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
             #if steps% print_every==0:
            valid_loss=0
            accuracy=0
            model.eval()
            with torch.no_grad():
                for inputs,labels in validation_loader:
                
                    inputs, labels= inputs.to(device), labels.to(device)
                    log_ps=model.forward(inputs)
                    batch_loss=criterion(log_ps, labels)
                    valid_loss+=batch_loss.item()
                    #calculating accuracy
                    ps = torch.exp(log_ps)
                    #top class, largest value in our prob.
                    top_p, top_class = ps.topk(1, dim=1)
                  #check for equality
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
            
        
         #turn our running loss back to zero 
        train_losses.append(running_loss/len(training_loader))
        valid_losses.append(valid_loss/len(validation_loader))
        train_accuracy.append(run_accuracy/len(training_loader))
        valid_accuracy.append(accuracy/len(validation_loader))
            
        
            
        running_loss=0
        model.train()
         
    return train_losses, valid_losses, valid_accuracy, train_accuracy
 #---------------------------------------------------------            
def testClassifier(model, criterion, validation_loader, current_device):
    model.eval()
    accuracy=0
        
    with torch.no_grad():
    #validation loop
        for inputs, labels in testloaders:
            inputs, labels= inputs.to(device), labels.to(device)
            log_ps=model.forward(inputs)
            batch_loss=criterion(log_ps, labels)
        
            #calculating accuracy
            ps = torch.exp(log_ps)
            #top class, largest value in our prob.
            top_p, top_class = ps.topk(1, dim=1)
            #check for equality
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            test_loss += batch_loss.item()
            #print out info   

        print(f"Test accuracy: {accuracy/len(testloaders):.3f}")
    return test_loss, accuracy
             
#---------------------------------------------------------
def main():
    args = parse_args()
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    current_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_transforms = transforms.Compose([
                                      transforms.Resize(256),
                                      transforms.RandomRotation(30),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(), transforms.Normalize(
                                      mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),])

    cost_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
                                     
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])
                                     ])




    train_datasets = datasets.ImageFolder(train_dir,transform=train_transforms) 
    cost_datasets = datasets.ImageFolder(valid_dir,transform=cost_transforms) 
    test_datasets = datasets.ImageFolder(test_dir,transform=test_transforms) 

    image_datasets = {"train": train_datasets, 
                   "validation": cost_datasets, 
                   "testing": test_datasets}


    trainloaders = torch.utils.data.DataLoader(train_datasets,batch_size=34, shuffle=True)
    costloaders = torch.utils.data.DataLoader(cost_datasets,batch_size=34)
    testloaders = torch.utils.data.DataLoader(test_datasets,batch_size=34)

    dataloaders = {"training": trainloaders,"validation": costloaders, "testing": testloaders}
             #------------------------------------------------
    model = models.vgg19(pretrained=True)
    model.name='vgg19'
    input_size=25088
    hidden_layers=[25088,1024]
    learning_rate=0.001
    output_size=102
    drop_out=0.002
    epochs=8
        #------------------------------------------------
    model.classifier=Classifier(input_size,output_size,hidden_layers,drop_out) 
    model.class_to_idx=train_datasets.class_to_idx
    params_to_update = []
    for param in model.parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    #define negative log likelihood loss
    criterion= nn.NLLLoss()
    #define optimizer 
    optimizer= optim.Adam(params_to_update, lr=0.002)
    model.to(current_device)
  
    with active_session():
        train_loss,valid_loss,valid_accuracy= training(model,epochs,criterion,optimizer,trainloaders,costloaders,current_device)           
             

    print("final result \n",
      f"Training loss: {train_loss:.3f}.. \n",
      f"Testing loss: {valid_loss:.3f}.. \n",
      f"Testing accuracy: {valid_accuracy:.3f}") 

    filename=saveCheckPoint(model)         
              
class Classifier(nn.Module):
    def __init__(self, input_size,output_size,hidden_layers,drop_out):
        super().__init__()
        self.hidden_layers=nn.ModuleList([nn.Linear(input_size,hidden_layers[0])])
        hlayers = zip(hidden_layers[:-1],hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(hinput,houtput) for hinput,houtput in hlayers])
        self.output = nn.Linear(hidden_layers[-1],output_size)
        self.dropout = nn.Dropout(p=drop_out)
        
    def forward(self,x):
        x = x.view(x.shape[0],-1)
        for layer in self.hidden_layers:
            x= self.dropout(F.relu(layer(x)))
            
        x= F.log_softmax(self.output(x),dim=1)
        return x
             
             
hidden_layers= [4096,1024]
input_size= 25088
output_size= 102
drop_out = 0.2
             
def saveCheckPoint(model):
    #checkpoint dictionary
    check_point= {
        'epochs': epochs,
        'optimizer': optimizer.state_dict,
        'arch': 'VGG19',
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'class_to_idx': train_datasets.class_to_idx}
    print('inside checkpoint')

    torch.save(check_point,'checkpoint.pth')
             
if __name__=="__main__":
    main()