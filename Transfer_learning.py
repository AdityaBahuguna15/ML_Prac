# Image Folder 
# Scheduler -> To change learning rate
# Transfer Learning
'''
Ml method, where a model developed for a first task, 
is reused as the the starting point for the 2nd task
eg: Train a model to classify birds and cats, and use the same model, 
modify it in last layers, to classify bees and dogs.

* basically changing the classification layer
'''
# Using Pretrained RESTNET 18-CNN
# trained on a million images on the ImageNet database
# 18 layers deep, can classify upto 1000 object categories

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

# import data
data_dir = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(class_names)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# One way of using Transfer Learning - Fine Tuning
model = models.resnet18(pretrained=True) # Available from torchvision models
num_features = model.fc.in_features

model.fc = nn.Linear(num_features, 2) # _, 2 classes (ant/bees) <- New last layer
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001)

# Scheduler -> Updates learning rate
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size= 7, gamma= 0.1) # Every 7 epochs, lr is multipled by 0.1
# for epoch in range(100):
#     train() # optimizer.step()
#     evaluate()
#     scheduler.step()

model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs= 20)


# [Faster] Option 2 For Transfer Learning: Freeze all layers except the very last one
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features

model.fc = nn.Linear(num_features, 2) # _, 2 classes (ant/bees) <- New last layer
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001)

# Scheduler -> Updates learning rate
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size= 7, gamma= 0.1) # Every 7 epochs, lr is multipled by 0.1

model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs= 20)

'''
model.state_dict() — a dictionary of all the model's learnable parameters
- It's a snapshot of every weight and bias in the model at that exact moment.

copy.deepcopy() — makes a completely independent copy

# without deepcopy — just a reference, points to same memory
best_model_wts = model.state_dict()
# if model weights change later, best_model_wts changes too ❌

# with deepcopy — completely separate copy in new memory
best_model_wts = copy.deepcopy(model.state_dict())
# model weights can change freely, best_model_wts stays frozen ✅

Together they save a frozen snapshot of the best weights seen so far. This is used for early stopping 
— if the model gets worse after epoch 15 but was best at epoch 12, you can restore epoch 12's weights:
# during training — save if best so far
if phase == 'val' and epoch_acc > best_acc:
    best_acc = epoch_acc
    best_model_wts = copy.deepcopy(model.state_dict())  # freeze this snapshot

# after training — restore the best snapshot
model.load_state_dict(best_model_wts)
'''