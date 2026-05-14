"""
Training Samples: Total number of rows in your training data
Batch Size: How many samples the model sees before updating its weights.
Eg: 455 samples, batch size = 32
Batch 1: samples 0-31 (32 samples) -> backward() -> update weights
Batch 2: samples 32-63 (32 samples) -> backward() -> update weights
and so on...

epoch: 1 forward and backward pass of ALL training samples
batch_size: number of training samples in one forward & backward pass
no. of iterations: no. of passes, each pass using [batch_size] number of samples
eg: 100 samples, batch_size =20 --> 100/20 = 5 iterations for 1 epoch

three aproaches:
Batch Size         Name             Beahvior
1                SGD (pure)         Updates every single sample - very noisy
all 455          Batch GD           One updates per epoch - trains very slow, memory heavy, gradient very accurate
32/64/128       Mini-batch GD       Best of both, bit noisy, uses little memory, learns faster, helps escape bad local minima

"""

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import time

class WineDataset(Dataset):
    def __init__(self): # Runs once, loads everything in memory
        # Data loading
        xy = np.loadtxt("./data.csv", delimiter=",", dtype = np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]) # n_samples, 1
        self.n_samples = xy.shape[0] # = 178
    
    def __getitem__(self, index): # lets you index the dataset like a list
        # dataset[0]
        return self.x[index], self.y[index]
    
    def __len__(self): # Tells dataloader how big the dataset is, uses it to know when one epoch is complete and to calc how many batches exist
        # len(dataset)
        return self.n_samples

dataset = WineDataset()
# first_data = dataset[2]
# features, labels = first_data
# print(features, labels)

# num_workers=2 causse issues on Windows for multiprocessing
# Either set it to 0, or wrap the code from dataset = WineDataset() in if __name__ == '__main__'
dataloader = DataLoader(dataset= dataset, batch_size= 4, shuffle=True, num_workers=0)

dataiter = iter(dataloader) # curretly each iter is 4 (same as batch_size)
# data = next(dataiter)
# features, labels = data
# print(features, labels)

# Training Loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward, backward, update 
        if (i+1) % 5 == 0:
            print(f'epoch{epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')

# torchvision.datasets.MNIST()
# fashion-mnist,, cifar, coco




# Additional Stuff: Comparing Multiprocessing times
# def measure_time(num_workers):
#     dataset = WineDataset()
#     dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=num_workers)
    
#     start = time.time()
    
#     for epoch in range(5):          # run a few epochs to see a real difference
#         for i, (features, labels) in enumerate(dataloader):
#             pass                    # simulates a training loop without actual training
    
#     end = time.time()
#     print(f"num_workers={num_workers} → {end - start:.4f} seconds")

# if __name__ == '__main__':
#     measure_time(num_workers=0)    # no multiprocessing
#     measure_time(num_workers=2)    # 2 workers
#     measure_time(num_workers=4)    # 4 workers
