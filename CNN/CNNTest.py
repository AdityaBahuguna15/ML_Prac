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
num_epochs = 0
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

def imshow(img):
    img = img /2 + 0.5 #unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

# get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Show images
# imshow(torchvision.utils.make_grid(images))

conv1 = nn.Conv2d(3,6,5)
pool = nn.MaxPool2d(2,2)
conv2 = nn.Conv2d(6,16,5)
#print(images.shape)
x = conv1(images)
#print(x.shape)
x = pool(x)
#print(x.shape)
x = conv2(x)
#print(x.shape)
x = pool(x)
#print(x.shape)