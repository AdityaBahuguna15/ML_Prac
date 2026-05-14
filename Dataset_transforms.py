# import torch
# import torchvision

# Pre-Written:
# dataset = torchvision.datasets.MNIST(
#     root= "./data", transform= torchvision.transforms.ToTensor())
# ^ using prexisting transforms on available datasets
# However expects data type to match what transform expects: PIL image or numy array of shape (H,W,C) <- Image format
# Will fail on wine CSV Data. Practically built for inage data. For tabular, we write our own transforms

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import time

class WineDataset(Dataset):
    def __init__(self, transform= None): # Runs once, loads everything in memory
        # Data loading
        xy = np.loadtxt("./data.csv", delimiter=",", dtype = np.float32, skiprows=1)
        self.n_samples = xy.shape[0] # = 178
        
        # We do NOT convert to tensor
        self.x =xy[:, 1:]
        self.y = xy[:, [0]] # n_samples, 1

        self.transform = transform
    
    def __getitem__(self, index): # lets you index the dataset like a list
        # dataset[0]
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def __len__(self): # Tells dataloader how big the dataset is, uses it to know when one epoch is complete and to calc how many batches exist
        # len(dataset)
        return self.n_samples

class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, sample): # lets you use an instance of the class as if it were a function
        inputs, target = sample
        inputs *= self.factor
        return inputs, target

# This is how u apply ur own transform to ur own dataset
dataset = WineDataset(transform=ToTensor()) # Converts to Tensor, if None, returns numpy
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)]) # Chains multiple transforms together into one pipeline
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))

'''
Transforms can be applied to PIL images, tensors, ndarrays, or custom data
during creation of the DataSet

complete list of built-in transforms: 
https://docs.pytorch.org/vision/0.8/transforms.html

On Images
---------
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale

On Tensors
----------
LinearTransformation, Normalize, RandomErasing

Conversion
----------
ToPILImage: from tensor or ndrarray
ToTensor : from numpy.ndarray or PILImage

Generic
-------
Use Lambda 

Custom
------
Write own class

Compose multiple Transforms
---------------------------
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])
'''
