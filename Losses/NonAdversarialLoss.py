from pytorch_msssim import ms_ssim
import torch
from torch import nn
from Configs import Global_Config

l1_criterion = torch.nn.L1Loss(reduction='mean')
l2_criterion = torch.nn.MSELoss(reduction='mean')


def rec_loss(attr_images, generated_images, a):
    ms_ssim_loss = 1 - ms_ssim(attr_images, generated_images, data_range=1, size_average=True)
    l1_loss_value = l1_criterion(attr_images, generated_images)
    return a * ms_ssim_loss + (1 - a) * l1_loss_value


def id_loss(encoded_input_image, encoded_generated_image):
    return l1_criterion(encoded_input_image, encoded_generated_image)


def landmark_loss(input_attr_lnd, output_lnd):
    loss = l2_criterion(input_attr_lnd, output_lnd)
    return loss


def l2_loss(attr_images, generated_images):
    loss = l2_criterion(attr_images, generated_images)
    return loss


# # Perceptual loss that uses a pretrained VGG network
# class VGGLoss(nn.Module):
#     def __init__(self):
#         super(VGGLoss, self).__init__()
#         self.vgg = VGG19().to(Global_Config.device)
#         self.criterion = nn.L1Loss()
#         self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
#
#     def forward(self, x, y):
#         x_vgg, y_vgg = self.vgg(x), self.vgg(y)
#         loss = 0
#         for i in range(len(x_vgg)):
#             loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
#         return loss

# def discriminator_loss(real_pred, fake_pred):
#     real_loss = F.softplus(-real_pred).mean()
#     fake_loss = F.softplus(fake_pred).mean()
#
#     return real_loss + fake_loss
