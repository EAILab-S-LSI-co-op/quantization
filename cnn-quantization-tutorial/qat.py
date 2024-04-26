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
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Load data
train_dataset = datasets.CIFAR10(root=data_path, train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.CIFAR10(root=data_path, train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Enable QAT
model.train()
model.qconfig = torch.ao.quantization.get_default_qat_qconfig(q_config)
model = torch.ao.quantization.QuantWrapper(model)
torch.ao.quantization.prepare_qat(model, inplace=True)

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
        
    if epoch > 3:
        model.apply(torch.ao.quantization.disable_observer)
    if epoch > 2:
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    # Prepare model for evaluation
    quantized_model = torch.ao.quantization.convert(model.eval(), inplace=False)
    quantized_model.eval()
    
# profile
measured_inference_latency = measure_inference_latency(quantized_model, device)
measured_acc = measure_accuracy(quantized_model, test_loader, device)
measured_size = get_size_of_model(quantized_model)

print("------------------------------------")
print("Model: ", model_name)
print("Method: QAT")
print("Inference latency: ", measured_inference_latency)
print("Accuracy: ", measured_acc)
print(f"Model size: {measured_size:.4f}MB")
print("------------------------------------")