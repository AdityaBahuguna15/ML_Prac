# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#    - forward pass: compute prediction and loss
#    - backward pass: gradients
#    - update weights
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets # pre-built datasets
from sklearn.preprocessing import StandardScaler # feature scaling
from sklearn.model_selection import train_test_split # data splitting

# 0) prepare data
# Dataset has 569 tumour measurements, Each tumour is labelled as malignant or benign
# The goal is to train a model to predict the label from measurements alone
bc = datasets.load_breast_cancer()
x, y = bc.data, bc.target

n_samples, n_features = x.shape

# Splitting data
# Splits the data where currently 80% is training, 20% (test_size) is final evaluation
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 1234)

# Scale
sc = StandardScaler() # Scale our features, feature to have zero mena and unit variance
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#convert to torch tensors
x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1) 
y_test = y_test.view(y_test.shape[0], 1) 

# 1) model
# f = wx + b, sigmoid function at the end
class LogisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1) # 30 input, 1 output
    
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x)) # Sigmoid fnct returns a value btwn 0-1
        return y_pred

model = LogisticRegression(n_features)

# 2) Loss and Optimizer
learning_rate = 0.03
criterion = nn.BCELoss() # Classes: only 2, eg: Outcomes: Yes/No, 0/1, malignant/benign
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# 3) Training loop
num_epochs = 100
for epoch in range(num_epochs):
    #Forward pass
    y_pred = model(x_train)
    loss = criterion(y_pred,y_train)

    #Backward pass
    loss.backward()

    #Update
    optimizer.step()

    #Zero grad
    optimizer.zero_grad()

    if(epoch+1) % 10 == 0:
        print(f'epoch{epoch+1}, loss = {loss.item():.4f}')

#Evaluation:
with torch.no_grad():
    y_pred = model(x_test)
    y_pred_cls = y_pred.round()
    acc = y_pred_cls.eq(y_test).sum() / float(y_test.shape[0]) # for every pred correct: +1
    print(f'accuracy = {acc:.4f}')

#Things to know:
"""
* Why we split data?
if you train and test on the same data, the model could just memorize the answers rather than learn the pattern. 
It would score 100% but fail on any new tumour it's never seen — useless in the real world.
The split enforces an honest evaluation. The model never sees x_test during training, 
so the final accuracy score is a genuine measure of how well it generalizes to new data.

* random_state=1234 
It is just a seed — without it the shuffle is different every run, 
giving you different accuracy scores each time. Setting it makes experiments reproducible.

* Why scale features
The 30 features have wildly different ranges:
- area mean:      1001.0   ← huge numbers
- smoothness mean:  0.118  ← tiny numbers
Without scaling, the model treats area mean as thousands of times more important than smoothness mean simply because its numbers 
are bigger — not because it's actually more useful. Gradient descent also converges much slower when features have mismatched scales.

* fit_transform vs transform
- fit_transform:
    fit — computes the mean and std of each feature from x_train
    transform — applies the scaling using those computed values
- transform:
    applies the same mean and std from x_train — does NOT recompute
This distinction matters a lot. If you called fit_transform on x_test too, it would compute new statistics from the test set 
which is a form of data leakage. The test set would be scaled differently than training, and you'd be peeking at test data during setup.


"""