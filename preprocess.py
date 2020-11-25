#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import os
import pandas as pd
import shutil

train_label = pd.read_csv('severstal-steel-defect-detection/train.csv')
train_dupli = train_label[train_label['ImageId'].duplicated(keep='first') == True]
# print(train_dupli.index)
train_label_deleted = train_label.drop(train_dupli.index)
# print(train_label_deleted)
# print(train_dupli.values)
# fp = open('dupli.txt','w')
# for i in train_dupli.values:
#     fp.write(i + '\n')
# fp.close()
# print(train_label['ImageId'].duplicated(keep='first').sum())

# X_train, X_val, y_train, y_val = train_test_split(train_label_deleted['ImageId'], train_label_deleted['ClassId'], test_size=0.2,shuffle=True,random_state=42)
# # print(len(X_train))
# # print(len(X_val))
#
#
path_img = 'severstal-steel-defect-detection/train_images'
image_name = os.listdir(path_img)
num_image = len(image_name)
for name in train_label_deleted['ImageId']:
    if name not in image_name:
        print(name)


X_train_list = list(X_train)
X_val_list = list(X_val)
num = 0
# fp = open('no_value.txt','w+')
for idx, file_name in enumerate(image_name):
    class_len = len(train_label[train_label['ImageId'] == image_name[idx]]['ClassId'].values)
    if( class_len != 0):
        if(file_name in X_train_list):
            pt = 'train'
        elif(file_name in X_val_list):
            pt = 'test'
        for i in range(class_len):
            path = os.path.join(pt, str(train_label[train_label['ImageId'] == image_name[idx]]['ClassId'].values[i]))
            shutil.copy(path_img + '/' + file_name, path)
    else:
        pass
        # fp.write(file_name + '\n')
        # print(idx, file_name)
# fp.close()

# print(num)
# num = 0
# for i in image_name:
#     if((train_label['ImageId']==i).sum()):
#         num += 1
# print(num)

num1 = 0
for i in range(1,5):
    path = os.path.join('train', str(i))
    num1 += len(os.listdir(path))
print(num1)

num2 = 0
for i in range(1,5):
    path = os.path.join('test', str(i))
    num2 += len(os.listdir(path))
print(num2)
print(num1 + num2)
