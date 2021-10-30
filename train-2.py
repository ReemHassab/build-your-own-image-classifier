# Imports here
#%matplotlib inline
#%config InlineBackend.figure_format ='retina'
import matplotlib.pyplot as plt
import torch 
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim 
import torch.nn.functional as F 
from collections import OrderedDict
from PIL import Image
import numpy as np
from torch.autograd import Variable

#------------------------------------------------
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
#------------------------------------------------
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
                                                           [0.229, 0.224, 0.225])
                                     ])
test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])
                                     ])



# TODO: Load the datasets with ImageFolder
train_datasets = datasets.ImageFolder(train_dir,transform=train_transforms) 
cost_datasets = datasets.ImageFolder(valid_dir,transform=cost_transforms) 
test_datasets = datasets.ImageFolder(test_dir,transform=test_transforms) 

image_datasets = {"train": train_datasets, 
                   "validation": cost_datasets, 
                   "testing": test_datasets}

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloaders = torch.utils.data.DataLoader(train_datasets,batch_size=34, shuffle=True)
costloaders = torch.utils.data.DataLoader(cost_datasets,batch_size=34)
testloaders = torch.utils.data.DataLoader(test_datasets,batch_size=34)

dataloaders= [trainloaders, costloaders, testloaders]
#------------------------------------------------
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
#------------------------------------------------
#initialize device to run on (GPU or CPU) 
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#load a pretraind network: chose vgg19 
model = models.vgg19(pretrained=True)
model
#turn off gradients for our model 
for param in model.parameters():
       param.requires_grad= False
#------------------------------------------------
import time
#defining a new feed-forward classifier using ReLu activation function
classifier= nn.Sequential(OrderedDict([
                                ('fc1',nn.Linear(25088,1024)),
                                 ('relu',nn.ReLU()),
                                  ('drop', nn.Dropout(0.2)),
                                  ('fc2',nn.Linear(1024,102)),
                                   ('output',nn.LogSoftmax(dim=1))
                                ]))
model.classifier=classifier
model
#------------------------------------------------
#define negative log likelihood loss
criterion= nn.NLLLoss()
#define optimizer 
optimizer= optim.Adam(model.classifier.parameters(), lr=0.002)
 
#cuda = torch.cuda.is_available()
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.cuda()

#trackings
epochs=8
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

for epoch in range (epochs):
    print('Epoch {}/{}'.format(epoch+1, epochs))
    print('-' * 10)
    
    # 2- looping through data
    for inputs,labels in trainloaders:
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
        for inputs,labels in costloaders:
                
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
    train_losses.append(running_loss/len(trainloaders))
    valid_losses.append(valid_loss/len(costloaders))
    train_accuracy.append(run_accuracy/len(trainloaders))
    valid_accuracy.append(accuracy/len(costloaders))
            
    print(
          f"Train loss: {running_loss/print_every:.3f}.. "
          f"Valid loss: {valid_loss/len(costloaders):.3f}.. "
          f"Train accuracy: {run_accuracy/print_every:.3f}.. "
          f"Valid accuracy: {accuracy/len(costloaders):.3f}.. ")       
            
    running_loss=0
    model.train()
                            
 
        
time_elapsed = time.time() - start
print("\nTotal time: {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))

#------------------------------------------------
#mapping classes to indices
model.class_to_idx = image_datasets['train'].class_to_idx

#checkpoint dictionary
check_point= {
            'epochs': epochs,
            'optimizer': optimizer.state_dict,
            'arch': 'VGG19',
            'classifier': model.classifier,
            'state_dict': model.state_dict(),
            'optimizer_dict': optimizer.state_dict(),
            'class_to_idx': train_datasets.class_to_idx}
    

torch.save(check_point,'checkpoint.pth')
#------------------------------------------------
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg19(pretrained=True)
    model.arch = checkpoint['arch']
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_dict'])
    
    
        
    return model
load_checkpoint('checkpoint.pth')
print(model)



       