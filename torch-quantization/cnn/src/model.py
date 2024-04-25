import torch
import torch.nn as nn
import torchvision.models as models


class CNN_1(nn.Module):
    def __init__(self):
        super(CNN_1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32 * 16 * 16, 10)
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 16 * 16)
        x = self.fc(x)
        return x
    
class CNN_2(nn.Module):
    def __init__(self):
        super(CNN_2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * 32 * 32, 10)

class CNN(nn.Module):
    def __init__(self, num_blocks):
        super(CNN, self).__init__()
        
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList([nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU()])
        for i in range(num_blocks):
            if i == 0:
                self.blocks.append(nn.Conv2d(32, 64, kernel_size=3, padding=1))
            else:
                self.blocks.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
            self.blocks.append(nn.ReLU())
            
        self.fc = nn.Linear(64 * 32 * 32, 10)
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.view(-1, 64 * 32 * 32)
        x = self.fc(x)
        return x
