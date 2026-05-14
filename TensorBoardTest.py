import torch
import torch.nn as nn # Layers, loss functions, activation functions
import torchvision 
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt
import sys
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/mnist2')
# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size = 784 #28x28 -> Images size, flatenning it to 1d tensor
hidden_size = 500
num_classes = 10 #-> 0-9
num_epochs = 1
batch_size = 64
learning_rate = 0.01

# MNIST (70000 handwritten digit images)
# torchvision.datasets.MNIST: a pre-built class that already implements __init__, __getitem__, and __len__
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, 
                transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset= train_dataset, 
                batch_size= batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset= test_dataset, 
                batch_size= batch_size, shuffle=False)

examples = iter(train_loader) # Batch Training
samples, labels = next(examples)
print(samples.shape, labels.shape) # batch_size = 100, 1 color channel (grayscale), image array: 28x28

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap= 'gray')
#plt.show()
img_grid = torchvision.utils.make_grid(samples)
writer.add_image('mnist_images', img_grid)
writer.close()
#sys.exit()

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device) # to.device for gpu usage

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate) # Adam adapts per weight, tracks history and converges faster than sgd

writer.add_graph(model, samples.reshape(-1, 28*28).to(device))
writer.close()
#sys.exit()

# Training loop
n_total_steps = len(train_loader)

running_loss = 0.0
running_correct = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 100, 1, 28, 28 <-batch, channel, H, W
        # ^ a 2D grid, but nn.Linear expects a 1D vector: 784
        images = images.reshape(-1, 28*28).to(device) # -1 means figure out this dimension auto
        labels = labels.to(device)

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predictions = torch.max(outputs.data, 1)
        running_correct += (predictions == labels).sum().item()
        if (i+1) % 100 == 0:
            print(f'epoch{epoch+1} / {num_epochs}, step {i+1}/ {n_total_steps}, loss = {loss.item():.4f}')
            writer.add_scalar('training_loss', running_loss / 100, epoch * n_total_steps + i)
            writer.add_scalar('accuracy', running_correct / 100, epoch * n_total_steps + i)
            running_loss = 0.0
            running_correct = 0

class_labels = []
class_preds = []
# Test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images) # *1

        # Value, Index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item() # Compares predictions against true labels

        class_probs_batch = [F.softmax(output, dim=0) for output in outputs]
        
        class_preds.append(class_probs_batch)
        class_labels.append(labels)
    
    class_preds = torch.cat([torch.stack(batch) for batch in class_preds])
    class_labels = torch.cat(class_labels)

    acc = 100 * n_correct / n_samples # Accuracy %
    print(f'accuracy = {acc}')

    classes = range(10)
    for i in classes:
        labels_i = class_labels == i
        preds_i = class_preds[:, i]
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
        writer.close()
