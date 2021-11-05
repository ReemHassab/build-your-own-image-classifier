# Imports here
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
import argparse
import time
import json

#initiatie variables with default values:

arch= 'vgg19'
hidden_units=1024
learning_rate= 0.001
epochs= 5
device= 'cpu' 
#------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',type= str, help='train on the data directory')
parser.add_argument('--save_dir',action='store', help='the path where to save the checkpoint')
parser.add_argument('--arch', default='vgg19',action='store',type=str, help='choose among 3 pretrained networks: vgg19, alexnet,densenet121')
parser.add_argument('--learning_rate',action='store', type=float, default=0.001, help='choose a float No. as the learning rate for the model')
parser.add_argument('--hidden_units',action='store', type=int, default=1024)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--gpu', action='store_true')
args = parser.parse_args()
#------------------------------------------------
#select parameters entered in command line:

if args.arch:
    arch=args.arch
    
if args.hidden_units:
    hidden_units= args.hidden_units
    
if args.learning_rate:
    learning_rate= args.learning_rate
    
if args.epochs:
    epochs= args.epochs
    
if args.gpu:
    #initialize device to run on (GPU or CPU) 
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    

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

valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])
                                     ])


# TODO: Load the datasets with ImageFolder
train_datasets = datasets.ImageFolder(train_dir,transform=train_transforms) 
valid_datasets = datasets.ImageFolder(valid_dir,transform=valid_transforms) 
 

image_datasets = {"train": train_datasets, 
                   "validation": valid_datasets 
                   }

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloaders = torch.utils.data.DataLoader(train_datasets,batch_size=34, shuffle=True)
validloaders = torch.utils.data.DataLoader(valid_datasets,batch_size=34)

dataloaders= [trainloaders, validloaders]

#------------------------------------------------
def create_model(arch='vgg19', hidden_units=1024,learning_rate=0.001):
   '''
   function builds model
   input: model architecture, hidden units, and learning rate
   output:model, criterion, optimizer, scheduler
   '''
   #load a pretraind network: chose vgg19 
   model = getattr(models, arch)(pretrained=True)
   in_features= model.classifier[0].in_features
   #model
   #turn off gradients for our model 
   for param in model.parameters():
       param.requires_grad= False
   #defining a new feed-forward classifier using ReLu activation function
   classifier= nn.Sequential(OrderedDict([
                                ('fc1',nn.Linear(in_features,hidden_units)),
                                 ('relu',nn.ReLU()),
                                  ('drop', nn.Dropout(0.2)),
                                  ('fc2',nn.Linear(1024,102)),
                                   ('output',nn.LogSoftmax(dim=1))
                                ]))
   model.classifier=classifier
   model
    #define negative log likelihood loss
   criterion= nn.NLLLoss()
   #define optimizer 
   optimizer= optim.Adam(model.classifier.parameters(), lr=0.002)
   #scheduler= lr_scheduler.stepLR(optimizer, step_size=2, gamma=0.1, epoch=-1)
    
   return model, criterion, optimizer
#------------------------------------------------
model, criterion, optimizer= create_model(arch, hidden_units, learning_rate)
print('-' * 10)
print('your model was successfully built!')
print('-' * 10)
#------------------------------------------------
def train_model(model, criterion, optimizer, epochs=4):
    '''
    function: trains the pretrained model and classifier on the image datasets, and validates.
    input:model, criterion, optimizer, epochs(default=4)
    output: trained model
    
   '''
    print_every=5
    running_loss=0
    accuracy=0
    step=0
    run_accuracy=0
    train_losses, valid_losses = [], []
    train_accuracy, valid_accuracy = [], []
    model.to(device)
    
    #define best model wheights and best accuracy
    

    print('begin training')
    start=time.time()
    #start training 

    # 1- training loop: looping through epochs
    model.train()

    for epoch in range (epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 10)
    
        # 2- looping through data
        for inputs,labels in trainloaders:
      
            
            #move input & label tensors to default device
            inputs, labels= inputs.to(device), labels.to(device)
            step+=1
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
        
            if step% print_every==0:
                valid_loss=0
                accuracy=0
                model.eval()
                with torch.no_grad():
                    for inputs,labels in validloaders:
                
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
                    
                    
            
        
         
    train_losses.append(running_loss/len(trainloaders))
    valid_losses.append(valid_loss/len(costloaders))
    train_accuracy.append(run_accuracy/len(trainloaders))
    valid_accuracy.append(accuracy/len(costloaders))
            
    print(
          f"Train loss: {running_loss/print_every:.3f}.. "
          f"Valid loss: {valid_loss/len(costloaders):.3f}.. "
          f"Train accuracy: {run_accuracy/print_every:.3f}.. "
          f"Valid accuracy: {accuracy/len(costloaders):.3f}.. ")       
    
    #turn our running loss back to zero 
    running_loss=0
    model.train()
                            
 
        
    time_elapsed = time.time() - start
    print("\nTotal time: {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))

    return model
#------------------------------------------------
trained_model= train_model(model,criterion,optimizer,epochs)

#------------------------------------------------

def save_model(trained_model):
    '''
    function saves our trained model
    input: model
    '''
    trained_model.class_to_idx = image_datasets['train'].class_to_idx
    trained_model.cpu()
    save_dir= ''
    #checkpoint dictionary
    check_point= {
            'epochs': epochs,
            'arch': arch,
            'hidden_units': hidden_units,
            'classifier': trained_model.classifier,
            'state_dict': ttrained_model.state_dict(),
            'class_to_idx': trained_model.class_to_idx
                }
    if args.save_dir:
        save_dir=args.save_dir
    else:
        save_dir='checkpoint.pth'
    torch.save(check_point, save_dir)
#------------------------------------------------
save_model(trained_model)
print('-' * 10)
print(trained_model)
print('your model has been successfully saved!')
print('-' * 10)

#------------------------------------------------







       
