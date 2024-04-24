import time
import resource
import torch

def profile(func, *args, **kwargs):
    start_time = time.time()

    result = func(*args, **kwargs)

    end_time = time.time() 
    exec_time = end_time - start_time

    return result, exec_time

def measure_accuracy(model, loader, device):
    model.eval()
    model.to(device)
    
    correct = 0
    for i, data in enumerate(loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        correct += torch.sum(predicted == labels).item()
    accuracy = correct / len(loader.dataset)
    return accuracy