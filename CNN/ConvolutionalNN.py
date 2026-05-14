# Using CIFAR-10 dataset
# Uses layers including CONV, activation (ReLU), POOL, Fully Connected (FC) layers
# FC is for actual classification done at the end (eg: Car: 80%, Truck: 12%, airplane: 5%....)

# Inital Image * Convolution Filter = Resulting Image (Smaller than original)

# Convolutional Layer:
'''
Top left 3x3 * w0 = Top left 1x1 and so on...
O O O O O
O O O O O    w0 w1 w2     O O O
O O O O O *  w3 w4 w5  =  O O O 
O O O O O    w6 w7 w8     O O O
O O O O O

Formula: (W-F + 2P)/S + 1 -> To calculate output size
ex: 5x5 input, 3x3 filter, padding=0, stride = 1
=> (5-3 + 0)/1 + 1 = 2/1 + 1 = 3 => 3x3

'''
# Max Pooling layers:
# Used to downsample an image by applying a max filter to sub-regions
# Used to reduce computational cost by reducing size of image, reducing no. of paramaters 
# needed to learn, and avoid overfitting and providing an abstracted form of the input
'''
012 020 | 30 00
008 012 | 02 00 
--------+-------  --------------->  020 30
034 070 | 37 04    (2x2 Max-Pool)   112 37
112 100 | 25 12

self.pool = nn.MaxPool2d(2, 2) # kernel size, stride
- Kernel Size: 2x2 blocks of pixel
- Stride: moves 2 pixels at a time(no overlap)
'''
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
num_epochs = 8
batch_size = 4
learning_rate = 0.001

# dataset has PILImage images of range [0,1]
# Transform them to Tensors of normalized range [-1,1]
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                transform=transform, download=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                transform=transform)

train_loader = torch.utils.data.DataLoader(dataset= train_dataset, 
                batch_size= batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset= test_dataset, 
                batch_size= batch_size, shuffle=False)

classes = ('plane', 'car', 'bird','cat', 'deer', 'dog', 
           'frog', 'horse', 'ship', 'truck')

# implement conv net
class ConvNet(nn.Module):
    def __init__(self): # *1
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # input, output, kernel_size (3 color channels
        self.pool = nn.MaxPool2d(2, 2) # kernel size, stride
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120) # Flattened to 1D and then compressed to 120
        self.fc2 = nn.Linear(120, 84) # compresses further from 120 -> 84
        self.fc3 = nn.Linear(84, 10) # compresses further from 84 -> 10
        
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5) # (4 batch_size, tensor is flattened here to 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ConvNet().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)

# Training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] -> 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device) 
        labels = labels.to(device)

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print(f'epoch{epoch+1} / {num_epochs}, step {i+1}/ {n_total_steps}, loss = {loss.item():.4f}')

# Test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        # max returns (Value, Index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predicted == labels).sum().item() # Compares predictions against true labels

        for i in range(batch_size): # accuracy of each class
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
    
    acc = 100 * n_correct / n_samples # Accuracy %
    print(f'Accuracy of Network: {acc} %')

    for i in range(10): # accuracy of each class
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')

'''
*1
Full picture:
Input (3x32x32)
      ↓
Conv2d(3→6, 5x5)    → (6x28x28)   learns basic features
      ↓
MaxPool2d(2x2)       → (6x14x14)   shrinks
      ↓
Conv2d(6→16, 5x5)   → (16x10x10)  learns complex features
      ↓
MaxPool2d(2x2)       → (16x5x5)   shrinks
      ↓
Flatten              → (400,)      goes from 2D → 1D
      ↓
Linear(400→120)      → (120,)      classify
      ↓
Linear(120→84)       → (84,)       classify
      ↓
Linear(84→10)        → (10,)       one score per class

Determining out_channels and kernel_size:
- Choosen by experimentation - no perfect value
- For Kernel Size: Odd numbers only -> so theres always a clear center pixel
- 3x3 for fine details, 7x7, used in first layer whe input images are large
- For out_channels: start small, double as you go deeper. Deeper layers get more 
  feature maps bcoz they detect more complex combinations of patterns

Why compress in fully connected layers, and when to stop?
After the conv layers you have a flat vector of 400 values — raw extracted features. 
The fully connected layers compress this into a final decision
- You stop at the number of classes (eg: 10 digits -> stops at 10)
- Dont shrink too fast (400 -> 10) -> Too aggressive, loses information
'''