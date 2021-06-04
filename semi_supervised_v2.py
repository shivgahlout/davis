import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import random
import glob
import sys
from torch.autograd import Variable
from skimage.io import imread, imsave
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import cv2
from metrics import *
from skimage.morphology import dilation
from skimage.morphology import disk
from torchvision.utils import make_grid
import pickle
from skimage import transform


''' MaskTrack Incorporated '''

####################################################################################

''' Functions to save the sample images'''

def save_grid(grid, name, cols=4, rows=4, Name=None):

    fig=plt.figure()
    n=-1
    for k in range(cols):
        
        for k_ in range(rows):
            n+=1
            plt.subplot(4,4, n+1)
            plt.imshow(grid[n])
            plt.axis('off')
            plt.axis('tight')
            plt.title(Name[n][0][:-4], fontsize=8)
    
    plt.savefig(name)

def save_input_images(grid, name, cols=4, rows=9, Name=None):

    grid=grid.numpy()
    grid=np.transpose(grid, (0,2,3,1))
 
    fig=plt.figure()
    n=-1
    for k in range(0,cols,2):
        for k_ in range(0,rows,2):
            n+=1
            plt.subplot(6,2, k+k_+1)
            plt.imshow((grid[n,:,:,:3]))
            plt.axis('off')
            plt.axis('tight')
            plt.title(Name[n][0][:-4], fontsize=8)
            plt.subplot(6,2, k+k_+2)
            plt.imshow(grid[n,:,:,3])
            plt.axis('off')
            plt.axis('tight')
            plt.title(Name[n][0][:-4], fontsize=8)
    
    plt.savefig(name)

####################################################################################################
''' Perform dilation and translation to generate the coarser maps from the current step. '''

selem = disk(5) ## for coarser
tform = transform.AffineTransform(translation = (10, -5)) ## for translation

def perform_dilation(maps, selem):
    maps=[dilation(transform.warp(I, tform.inverse), selem) for I in maps]
    return np.array(maps)

######################################################################################################

''' Merge the input image and coarser map from the previous step '''

def merge_maps(images, maps, axis, save_plot=False, Name=None):
    
    if save_plot:
        save_grid(maps, 'map_grid.png', Name=Name)

    maps=perform_dilation(maps, selem)

    if save_plot:
        save_grid(maps, 'dilated_grid.png', Name=Name)

    maps=torch.unsqueeze(torch.tensor(maps),1)

    if save_plot:
        save_input_images(torch.cat((images, maps), axis), 'inputs.png', Name=Name)

    
    return torch.cat((images, maps), axis)
    
######################################################################################

'''MaskTrack Dataloader. Loading images at the class level. No augmentations yet.'''

class BasicDataset(Dataset):
    def __init__(self,class_dict, image_dict, phase='train'):
        self.class_dict=class_dict
        self.image_dict=image_dict
        self.image_size=224
        self.phase=phase
     

        if self.phase =='val' or self.phase=='test':
            
            self.custom_transform = transforms.Compose([ transforms.ToPILImage(),
                                                    transforms.Resize([self.image_size,self.image_size]),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) ])

        else:
            self.custom_transform = transforms.Compose([ transforms.ToPILImage(),
                                                   transforms.Resize([self.image_size,self.image_size]),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) ])



        # self.num_images = len(self.images)
        # print(self.num_images)

        self.num_classes=len(self.class_dict)

    def __len__(self):
        return self.num_classes

    def __getitem__(self, idx):

        ##load image

        class_name=self.class_dict[idx]
        images=self.image_dict[class_name]

        class_bag=torch.zeros(len(images), 3, self.image_size, self.image_size)
        label_bag=np.zeros((len(images), self.image_size, self.image_size))

        name_bags=[]

        counter=-1
        for image_name in images:
            counter+=1
            image=imread('../DAVIS2017_dataset/'+ self.phase + '_orig/data/'+ class_name+'/' +image_name)
            image=self.custom_transform(image)
            class_bag[counter]=image

            label=imread('../DAVIS2017_dataset/GT_binary/'+ class_name+'/' +image_name[:-3]+'png')
            label = cv2.resize(label, (self.image_size, self.image_size), interpolation = cv2.INTER_NEAREST)
            label[label==255]=1

            label_bag[counter]=label

            name_bags.append(class_name+'_'+image_name)

        return class_bag, label_bag, name_bags



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

############################################################################

''' hyperparameters'''

batch_size=1
aux_batch_size=20
num_classes=1
learning_rate=.0005
total_epochs=20
num_classes=1
criterion=nn.BCEWithLogitsLoss()


## create train dataset loader
with open('/mnt/drive2/shiv/DAVIS2017_dataset/train_dicts/class_dict.pickle', 'rb') as f:
        train_class_dict = pickle.load(f) 
with open('/mnt/drive2/shiv/DAVIS2017_dataset/train_dicts/image_dict.pickle', 'rb') as f:
        train_images_dict = pickle.load(f)

train_dataset = BasicDataset(train_class_dict,train_images_dict)
train_dataset_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True, num_workers=8)

## create val dataset loader
with open('/mnt/drive2/shiv/DAVIS2017_dataset/val_dicts/class_dict.pickle', 'rb') as f:
        val_class_dict = pickle.load(f) 
with open('/mnt/drive2/shiv/DAVIS2017_dataset/val_dicts/image_dict.pickle', 'rb') as f:
        val_images_dict = pickle.load(f)
val_dataset = BasicDataset(val_class_dict,val_images_dict, phase='val')
val_dataset_loader = DataLoader(val_dataset,batch_size=batch_size, shuffle=False, num_workers=8)


''' using a pretraiined inceptionV3 model'''
model = models.segmentation.deeplabv3_resnet101(pretrained=True)

''' modify according to the number of classes and input channels'''
model.backbone.conv1=nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.classifier = DeepLabHead(2048, num_classes)
model.cuda()
# model=nn.DataParallel(model)

''' Adam optimizer. No learning rate scheduler yet '''
optimizer=optim.Adam(model.parameters(), lr=learning_rate)


''' Training and validation '''

for epoch in range(total_epochs):
    train_losses = AverageMeter()
    train_iou = AverageMeter()
    model.train()
    for j, (image_bag, label_bag, name_bag) in enumerate(train_dataset_loader):
        # print(label_name)
        image_bag=image_bag.squeeze()
        label_bag=label_bag.squeeze()
        # print(name_bag[0][0][:-4])
        # print(image_bag.shape)
        k_=-1
        for k in range(0, image_bag.shape[0], aux_batch_size):
            # 
            k_+=1
            images=image_bag[k_*aux_batch_size:(k_+1)*aux_batch_size] if (k_+1)*aux_batch_size < image_bag.shape[0] else image_bag[k_*aux_batch_size:]
            labels=label_bag[k_*aux_batch_size:(k_+1)*aux_batch_size] if (k_+1)*aux_batch_size < label_bag.shape[0] else label_bag[k_*aux_batch_size:]

            # print(images.shape)
            if images.shape[0]==1:
                images=torch.cat((images, images), dim=0)
                labels=torch.cat((labels, labels), dim=0)

            # print(images.shape)

            # if j==0 or j==len(train_dataset)//batch_size:
            if epoch==0 and j==0 and k==0:
                images=merge_maps(images, labels,1, True, name_bag)
            else:
                images=merge_maps(images, labels,1)

      

            images=Variable(images.float().cuda(), requires_grad=True)
            labels=Variable(labels.float().cuda(), requires_grad=False)
            ## Inception-v3 provides two outputs. 
            model_outputs=model(images)['out'].squeeze()

            
            optimizer.zero_grad()

            # print(model_outputs.shape)
            # print(labels.shape)

            loss=criterion(model_outputs, labels)
            train_losses.update(loss.item(), images.size(0))

            # with amp.scale_loss(loss, optimizer) as scaled_loss:
                # scaled_loss.backward()

            loss.backward()
            optimizer.step()

            # previous_predictions=torch.ge(torch.sigmoid(model_outputs), 0.5).float().data.cpu().numpy()
            train_iou.update(cal_iou(labels.data.cpu().numpy().ravel(),\
            torch.ge(torch.sigmoid(model_outputs), 0.5).float().data.cpu().numpy().ravel()), images.size(0))
    
    val_predictions=[]
    val_labels=[]
    val_losses = AverageMeter()
    # val_acc=AverageMeter()
    val_iou=AverageMeter()
    model.eval()
    for j, (image_bag, label_bag, name_bag) in enumerate(val_dataset_loader):
        image_bag=image_bag.squeeze()
        label_bag=label_bag.squeeze()
        k_=-1
        for k in range(0, image_bag.shape[0], aux_batch_size+1):
            k_+=1
            images=image_bag[k_*aux_batch_size:(k_+1)*aux_batch_size] if (k_+1)*aux_batch_size < image_bag.shape[0] else image_bag[k_*aux_batch_size:]
            labels=label_bag[k_*aux_batch_size:(k_+1)*aux_batch_size] if (k_+1)*aux_batch_size < label_bag.shape[0] else label_bag[k_*aux_batch_size:]
            # print(images.shape)
            # print(labels.shape)
            if images.shape[0]==1:
                images=torch.cat((images, images), dim=0)
                labels=torch.cat((labels, labels), dim=0)
            with torch.no_grad():
                # if j==0 or j==len(val_dataset)//batch_size:
                images=merge_maps(images, labels,1)
                # else:
                #     images=merge_maps(images, previous_predictions,1)
                images=Variable(images.float().cuda())
                labels=Variable(labels.float().cuda())
                model_outputs=model(images)['out'].squeeze()
                loss=criterion(model_outputs, labels)
                val_losses.update(loss.item(), images.size(0))
                val_iou.update(cal_iou(labels.data.cpu().numpy().ravel(), \
                    torch.ge(torch.sigmoid(model_outputs), 0.5).float().data.cpu().numpy().ravel()), images.size(0))
                # previous_predictions=torch.ge(torch.sigmoid(model_outputs), 0.5).float().data.cpu().numpy()

    
    print('*'*20)
    print('epoch: {}, train_loss {} valid_loss: {} '.format(epoch, train_losses.avg, val_losses.avg))
    print('epoch: {}, train_iou {} valid_iou: {} '.format(epoch, train_iou.avg, val_iou.avg))
    print('*'*20)
      
    torch.save(model.state_dict(), 'weights/semi_supervised')

        
        





        
        




