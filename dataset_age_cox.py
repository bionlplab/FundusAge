import pickle
import os
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import numpy as np
import time
import copy
import torchvision.transforms as transforms
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    #'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, labels, groups, ages, transform=None):
        'Initialization'
        self.groups = groups
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform
        self.ages = ages
    def __len__(self):
       # 'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
       # 'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
            
        X = Image.open('/prj0129/mil4012/AREDS/AMD_224/' + ID[:-4] + '.jpg')
        if self.transform:
             X = self.transform(X)
        
        y = self.labels[index]
        event = torch.tensor([1], dtype=torch.int32)
        group = self.groups[index]
        
        age = self.ages[index]
        #print(type(age),type(y))
        #print(age,y)
        age = torch.FloatTensor(age)
        # return X, torch.tensor(y),torch.tensor(group)
        # return X, torch.tensor(y),group
        return X, torch.FloatTensor(y), event, group, age, ID