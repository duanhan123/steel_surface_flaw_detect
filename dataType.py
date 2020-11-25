#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torch.autograd import Variable
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image
#%%
image_size = (256, 1600, 3)
num_classes = 4
num_epochs = 1
batch_size = 8
LR = 0.001
train_label = pd.read_csv('severstal-steel-defect-detection/train.csv')


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataSet(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader):
        self.image_files = []
        for i in range(1,5):
            for x in os.scandir(os.path.join(root, str(i))):
                label = train_label[train_label['ImageId'] == x.name]['ClassId']
                self.image_files.append((x.path, list(label)[0]-1))
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        image_path, label = self.image_files[index]
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_files)


class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,16,5,1,2), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(nn.Conv2d(16,32,5,1,2), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.out = nn.Linear(32*64*400, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x

