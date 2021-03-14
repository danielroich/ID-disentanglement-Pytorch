import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms


class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        full_inception = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True,
                                        aux_logits=False, init_weights=False)

        removed = list(full_inception.children())[:-1]
        self.model = nn.Sequential(*removed)

    def forward(self, data):

        return self.model(data)
