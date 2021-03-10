import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms


class Encoder_Attribute(nn.Module):
    def __init__(self,  pretrained=False):
        super(Encoder_Attribute, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=pretrained, aux_logits=False, init_weights=False)
        self.model.fc = Identity()
        self.meta = {'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225],
                     'imageSize': [299, 299, 3]}

    def forward(self, data):
        return self.model(data)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x