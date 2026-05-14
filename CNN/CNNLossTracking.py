'''
Epoch test losses  → tells you HOW training is going (the graph)
Final evaluation   → tells you HOW GOOD the final model is (the result)

To be aware of:
One thing to be aware of
Because the final evaluation runs on the same test set you've been monitoring during training, 
there's a subtle risk — if you use the test loss during epochs to make decisions (like early stopping or tuning hyperparameters), 
the test set is technically influencing your training process. In production you'd have a third split:
train set      → model learns from this
validation set → monitor loss during training, make decisions
test set       → touched exactly once at the very end for final result

model.train() and model.eval():
They are mode switches that tell the model how to behave during the forward pass. They don't start training or 
testing themselves — they just change the internal behaviour of certain layers
Training loop:
    model.train() → dropout active → neurons randomly dropped → prevents memorization

Test/eval loop:
    model.eval()  → dropout off   → all neurons used → stable accurate predictions

Dropout:
model.train() — dropout is ACTIVE
randomly zeros out 30% of neurons each forward pass
x = [1.2, 0.8, 3.1, 0.5, 2.2]
  → [0.0, 0.8, 0.0, 0.5, 2.2]   ← some randomly killed

model.eval() — dropout is DISABLED
all neurons pass through unchanged
x = [1.2, 0.8, 3.1, 0.5, 2.2]
  → [1.2, 0.8, 3.1, 0.5, 2.2]   ← nothing dropped
'''
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 50
batch_size = 4
learning_rate = 0.001

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                transform=transform, download=True)
test_dataset  = torchvision.datasets.CIFAR10(root='./data', train=False,
                transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model     = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# track losses
train_losses = []
test_losses  = []

n_total_steps = len(train_loader)

for epoch in range(num_epochs):

    # training
    model.train() # good practice to put, doesnt do anything yet since we dont have Dropout or BatchNorm in ConvNet
    epoch_train_loss = 0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss    = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()

        if (i+1) % 2000 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

    train_losses.append(epoch_train_loss / n_total_steps)  # avg train loss

    # --- test loss per epoch ---
    model.eval()
    epoch_test_loss = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images  = images.to(device)
            labels  = labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            epoch_test_loss += loss.item()

    test_losses.append(epoch_test_loss / len(test_loader))  # avg test loss
    print(f'epoch {epoch+1}/{num_epochs}  train: {train_losses[-1]:.4f}  test: {test_losses[-1]:.4f}')

# final evaluation
model.eval() # good practice to put, doesnt do anything yet since we dont have Dropout or BatchNorm in ConvNet
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred  = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100 * n_correct / n_samples
    print(f'\nAccuracy of Network: {acc:.2f}%')
    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]:>6}: {acc:.1f}%')

# plot
plt.figure(figsize=(9, 5))
plt.plot(range(1, num_epochs+1), train_losses, label='train loss')
plt.plot(range(1, num_epochs+1), test_losses,  label='test loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CIFAR10 — Training vs Test Loss')
plt.xticks(range(1, num_epochs+1))
plt.legend()
plt.tight_layout()
plt.show()