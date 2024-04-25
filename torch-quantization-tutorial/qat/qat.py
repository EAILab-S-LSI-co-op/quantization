import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision import models
from torch.utils.data import DataLoader

from src.util import *
from src.vgg import *
from src.option import quantize_vgg_arguments

config = quantize_vgg_arguments()

model_name = config.model_name
data_path = config.data_path
model_path = config.model_path
device = 'cpu' # torch quantization does not support CUDA yet
    
# Load pre-trained model
model = eval(f"{model_name}()").to(device)
model.load_state_dict(torch.load(f"{model_path}/{model_name}.pth"))
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# Load data
train_dataset = datasets.CIFAR10(root=data_path, train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.CIFAR10(root=data_path, train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Fuse modules
# model.eval() # model must be set to eval for fusion to work
# model = torch.quantization.fuse_modules(model, [['net.conv1', 'net.bn1', 'net.relu']])

# Enable QAT
model.train()
model.qconfig = torch.quantization.get_default_qat_qconfig() # 'qnnpack'
model = torch.quantization.QuantWrapper(model)
torch.quantization.prepare_qat(model, inplace=True)

model.to(device)

# Train
num_epochs = 1
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

# Prepare model for evaluation
model.eval()
quantized_model = torch.quantization.convert(model, inplace=False)

# profile
x = torch.randn(1, 3, 32, 32)
result, exec_time, _ = profile(quantized_model, x)
print(f"Time taken (seconds): {exec_time:.4f}")


# Save quantized model
accuracy = measure_accuracy(quantized_model, test_loader, device)
print(f"Accuracy: {accuracy}")
torch.save(quantized_model.state_dict(), f"{model_path}/quantized_{model_name}.pth")