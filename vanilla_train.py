import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
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
from apex import amp
from metrics import *


''' Vanilla supervised training. Will be modified for the semi-supervised case'''

'''Vanilla Dataloader. No augmentations yet. To be modified for the semi-supervised case'''


class BasicDataset(Dataset):
    def __init__(self,folder_names, validation = False):
        self.images = list()
        self.image_size=224
        # print(folder_names)
        for fold in os.listdir(folder_names):
                print(fold)
                self.images += glob.glob(os.path.join(folder_names,fold)+"/*/*")

        if not validation:
            
            self.custom_transform = transforms.Compose([ transforms.ToPILImage(),
                                                    transforms.Resize([self.image_size,self.image_size]),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) ])

        else:
            self.custom_transform = transforms.Compose([ transforms.ToPILImage(),
                                                   transforms.Resize([self.image_size,self.image_size]),
                                                    transforms.ToTensor().
                                                    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) ])



        self.num_images = len(self.images)
        print(self.num_images)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):

        ##load image

        image_name = self.images[idx]
        image=imread(image_name)

        image=self.custom_transform(image)

        ##load GT and create binary map

        label_name='../DAVIS2017_dataset/GT/'+image_name.split('/')[-2]+'/'+ image_name.split('/')[-1][:-3]+'png'
        label=imread(label_name)
        label = cv2.resize(label, (self.image_size, self.image_size), interpolation = cv2.INTER_LINEAR)
       
        if len(label.shape)>2:
            label=np.sum(label[:,:,:3],2).astype('uint8')
        label[label!=0]=1

        return image, label



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



batch_size=32
num_classes=1
learning_rate=.0001
total_epochs=20
num_classes=1
criterion=nn.BCEWithLogitsLoss()


## create train dataset loader
train_folders='../DAVIS2017_dataset/train_orig/'
train_dataset = BasicDataset(train_folders)
train_dataset_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True, num_workers=8)

## create val dataset loader
val_folders='../DAVIS2017_dataset/val_orig/'
val_dataset = BasicDataset(val_folders)
val_dataset_loader = DataLoader(val_dataset,batch_size=batch_size, shuffle=True, num_workers=8)


## using a pretraiined inceptionV3 model
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
## modify according to the number of classes
model.classifier = DeepLabHead(2048, num_classes)
model.cuda()
optimizer=optim.Adam(model.parameters(), lr=learning_rate)
''' Using amp for mixed precision training '''
model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)


for epoch in range(total_epochs):
    train_losses = AverageMeter()
    train_iou = AverageMeter()
    model.train()
    for j, (images, labels) in enumerate(train_dataset_loader):

        images=Variable(images.cuda(), requires_grad=True)
        labels=Variable(labels.float().cuda(), requires_grad=False)
        ## Inception-v3 provides two outputs. 
        model_outputs=model(images)['out'].squeeze()

        optimizer.zero_grad()

        loss=criterion(model_outputs, labels)
        train_losses.update(loss.item(), images.size(0))

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()
        train_iou.update(cal_iou(labels.data.cpu().numpy().ravel(),\
        torch.ge(torch.sigmoid(model_outputs), 0.5).float().data.cpu().numpy().ravel()), images.size(0))
    
    val_predictions=[]
    val_labels=[]
    val_losses = AverageMeter()
    # val_acc=AverageMeter()
    val_iou=AverageMeter()
    model.eval()
    for j, (images, labels) in enumerate(val_dataset_loader):
        with torch.no_grad():
            images=Variable(images.cuda())
            labels=Variable(labels.float().cuda())
            model_outputs=model(images)['out'].squeeze()
            loss=criterion(model_outputs, labels)
            val_losses.update(loss.item(), images.size(0))
            val_iou.update(cal_iou(labels.data.cpu().numpy().ravel(), \
                torch.ge(torch.sigmoid(model_outputs), 0.5).float().data.cpu().numpy().ravel()), images.size(0))

    
    print('*'*20)
    print('epoch: {}, train_loss {} valid_loss: {} '.format(epoch, train_losses.avg, val_losses.avg))
    print('epoch: {}, train_iou {} valid_iou: {} '.format(epoch, train_iou.avg, val_iou.avg))
    print('*'*20)
      
    torch.save(model.state_dict(), 'weights/supervised_vanilla')

        
        





        
        




