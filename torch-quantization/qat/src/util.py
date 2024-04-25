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


def measure_inference_latency(model, 
                              device,
                              input_size=(1, 3, 32, 32),
                              num_samples=100,
                              num_warmups=10):
    model.to(device)
    model.eval()

    x = torch.rand(size=input_size).to(device)

    with torch.no_grad():
        for _ in range(num_warmups):
            _ = model(x)
    torch.cuda.synchronize()

    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_samples):
            _ = model(x)
            torch.cuda.synchronize()
        end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time = elapsed_time / num_samples

    return elapsed_time

def measure_model_size(model):
        
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = param_size + buffer_size
    return size_all_mb