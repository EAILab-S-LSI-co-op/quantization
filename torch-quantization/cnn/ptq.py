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
q_config = config.qconfig
device = 'cpu' 
    
# Load pre-trained model
# model = eval(f"{model_name}()").to(device)
model = vgg(model_name).to(device)
model.load_state_dict(torch.load(f"{model_path}/{model_name}.pth"))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# Load data
train_dataset = datasets.CIFAR10(root=data_path, train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.CIFAR10(root=data_path, train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Enable QAT
model.eval()
model.qconfig = torch.quantization.get_default_qat_qconfig(q_config)
model = torch.quantization.QuantWrapper(model)
torch.ao.quantization.prepare(model, inplace=True)

# Calibration
for data in train_loader:
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    model(inputs)    

quantized_model = torch.quantization.convert(model.eval(), inplace=False)
torch.save(quantized_model.state_dict(), f"{model_path}/ptq_static_{model_name}.pth")