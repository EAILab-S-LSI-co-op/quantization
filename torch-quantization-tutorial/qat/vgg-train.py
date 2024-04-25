import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import CNN
from src.vgg import *
from src.util import *
from src.option import train_vgg_arguments

config = train_vgg_arguments()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = config.model_name
model_path = config.model_path
data_path = config.data_path
if not os.path.exists(model_path):
    os.makedirs(model_path)

# Load pre-trained model
model = eval(f"{model_name}()").to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Load data
train_dataset = datasets.CIFAR10(root=data_path, train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.CIFAR10(root=data_path, train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Train
num_epochs = 150
for epoch in tqdm(range(num_epochs)):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        model.train()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
    if epoch % 10 == 9:
        accuracy = measure_accuracy(model, test_loader, device)
        torch.save(model.state_dict(), f"{model_path}/{model_name}.pth")
        print(f"Epoch: {epoch+1}, Accuracy: {accuracy}")

accuracy = measure_accuracy(model, test_loader, device)
print(f"Accuracy: {accuracy}")
torch.save(model.state_dict(), f"{model_path}/{model_name}.pth")