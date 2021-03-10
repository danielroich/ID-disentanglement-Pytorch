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

        # full_inception = torchvision.models.inception_v3(pretrained=False, aux_logits = False)

        self.meta = {'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225],
                     'imageSize': [299, 299, 3]}

        removed = list(full_inception.children())[:-1]
        self.model = nn.Sequential(*removed)

    def forward(self, data):
        return self.model(data)
