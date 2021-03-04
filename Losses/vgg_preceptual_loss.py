import torch

from torchvision.models import vgg16
from fastai.vision.learner import create_body
from fastai.callbacks.hooks import hook_outputs

class PerceptualLoss(nn.Module):
    def __init__(self, backbone=vgg16_backbone):
        super(PerceptualLoss, self).__init__()
        self.vgg = vgg16_backbone
        self.save_feature_idxs = [3, 8, 15, 22, 29]  # All layers before a MaxPool.
        self.hooks = hook_outputs([backbone[i] for i in self.save_feature_idxs])
        self.criterion = nn.L1Loss()
        self.weights = [1.0, 1.0 / 2, 1.0 / 4, 1.0 / 2, 1.0]  # information from earlier and later layers of the image

    def forward(self, real, fake):

        #         import pdb
        #         pdb.set_trace()

        with torch.no_grad():
            self.vgg(real)
            real_features = [feat.stored for feat in self.hooks]
        #             self.remove()

        with torch.no_grad():
            self.vgg(fake)
            fake_features = [feat.stored for feat in self.hooks]
        #             self.remove()

        loss = 0
        for i in range(len(real_features)):
            loss += self.weights[i] * self.criterion(real_features[i], fake_features[i])

        #         self.remove()
        return loss

    def remove(self):
        for hook in self.hooks:
            hook.remove()