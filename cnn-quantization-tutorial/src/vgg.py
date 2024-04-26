import torch
import torch.nn as nn

_cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def _make_layers(cfg):
    layers = []
    in_channels = 3
    for layer_cfg in cfg:
        if layer_cfg == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels=in_channels,
                                    out_channels=layer_cfg,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=True))
            layers.append(nn.BatchNorm2d(num_features=layer_cfg))
            layers.append(nn.ReLU(inplace=True))
            in_channels = layer_cfg
    return nn.Sequential(*layers)


class vgg(nn.Module):
    """
    vgg module for 3x32x32 input, 10 classes
    """

    def __init__(self, name):
        super(vgg, self).__init__()
        cfg = _cfg[name]
        self.layers = _make_layers(cfg)
        flatten_features = 512
        self.fc1 = nn.Linear(flatten_features, 10)

    def forward(self, x):
        x = self.layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        return x


def vgg11():
    return vgg('vgg11')


def vgg13():
    return vgg('vgg13')


def vgg16():
    return vgg('vgg16')


def vgg19():
    return vgg('vgg19')


# class ConvBNReLU(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
#         super(ConvBNReLU, self).__init__(
#             nn.Conv2d(in_channels=in_channels,
#                       out_channels=out_channels,
#                       kernel_size=kernel_size,
#                       stride=stride,
#                       padding=padding,
#                       bias=True),
#             nn.BatchNorm2d(num_features=out_channels),
#             nn.ReLU(inplace=True)
#         )

# def _make_layers(cfg):
#     layers = []
#     in_channels = 3
#     for layer_cfg in cfg:
#         if layer_cfg == 'M':
#             layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
#         else:
#             layers.append(ConvBNReLU(in_channels, layer_cfg))
#             in_channels = layer_cfg
#     return nn.Sequential(*layers)

# class vgg(nn.Module):
#     """
#     vgg module for 3x32x32 input, 10 classes
#     """
#     def __init__(self, name):
#         super(vgg, self).__init__()
#         cfg = _cfg[name]
#         self.layers = _make_layers(cfg)
#         flatten_features = 512
#         self.fc1 = nn.Linear(flatten_features, 10)

#     def forward(self, x):
#         x = self.layers(x)
#         x = x.reshape(x.size(0), -1)
#         x = self.fc1(x)
#         return x

#     def fuse(self):
#         for m in self.modules():
#             if type(m) == ConvBNReLU:
#                 torch.ao.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
    