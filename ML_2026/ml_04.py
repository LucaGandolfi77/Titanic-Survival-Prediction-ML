'''
from torch.utils.data import DataLoader
train_loader= DataLoader(dataset, batch_size=32, shuffle=True)

from torchvision.datasets import FashionMNIST
train_data= FashionMNIST(
root="data",
train=True,
download=True
)

from torchvision import transforms
transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.5,), (0.5,))
])

import torch.nn as nn
linear = nn.Linear(in_features, out_features)
# x must have shape (batch_size, in_features)
y = linear(x)

import torch.nn as nn
sigmoid= nn.Sigmoid()
y = sigmoid(y)

import torch.nn as nn
relu = nn.ReLU()
y = relu(y)

import torch.nn as nn
criterion = nn.CrossEntropyLoss()
loss = criterion(predictions, targets)

from torch.optim import SGD
opt= SGD(model.parameters(), lr=0.01)

import torch.nn as nn
criterion = nn.CrossEntropyLoss()
loss = criterion(predictions, targets)

model.train()
for inputs, labels in train_loader:
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
... # Compute metrics

model.eval()
with torch.no_grad():
for inputs, labels in dev_loader:
outputs = model(inputs)
loss = criterion(outputs, labels)
... # Compute metrics
'''
