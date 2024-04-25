import time
import resource
import torch

def profile(func, *args, **kwargs):
    start_time = time.time()
    start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    result = func(*args, **kwargs)

    end_time = time.time() 
    end_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    exec_time = end_time - start_time
    mem_usage = end_mem - start_mem

    return result, exec_time, mem_usage



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