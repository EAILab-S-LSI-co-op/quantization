import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision import models
from torch.utils.data import DataLoader

from src.option import *
from src.util import *

config = parse_arguments()

model_name = config.m
device = 'cpu'
    
# Load pre-trained model
model = eval(f"models.{model_name}()")
model.load_state_dict(torch.load(f"./models/{model_name}.pth"))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Load data
train_dataset = datasets.CIFAR10(root='/workspace/shared/data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.CIFAR10(root='/workspace/shared/data', train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# model.eval() # model must be set to eval for fusion to work
# model = torch.quantization.fuse_modules(model, [['model.conv1', 'model.relu']])

# Enable QAT
model.train()
model.qconfig = torch.quantization.get_default_qat_qconfig() # 'qnnpack'
model = torch.quantization.QuantWrapper(model)
torch.quantization.prepare_qat(model, inplace=True)

model.to(device)
# Train
num_epochs = 5
for epoch in range(num_epochs):
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
        if i % 100 == 99:
            print(f"Epoch {epoch+1}, Batch {i+1}: Loss {running_loss / 100}")
            running_loss = 0.0
            
            model.eval()
            quantized_model = torch.quantization.convert(model, inplace=False)
            # model test
            correct = 0
            for i, data in enumerate(test_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = quantized_model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                correct += torch.sum(predicted == labels).item()
            accuracy = correct / len(test_dataset)
            print(f"Accuracy: {accuracy}")

            # profile
            x = torch.randn(1, 3, 224, 224)
            result, exec_time, mem_usage = profile(quantized_model, x)
            print("Time taken (seconds):", exec_time)
            print("Memory used (KB):", mem_usage)
            

# Prepare model for evaluation
# model.eval()
# torch.quantization.convert(model, inplace=True); 


# Save quantized model
# torch.save(model.state_dict(), "quantized_model.pth")