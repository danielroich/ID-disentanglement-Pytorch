from pytorch_msssim import ms_ssim
import torch

l1_criterion = torch.nn.L1Loss(reduction='mean')


def rec_loss(attr_images, generated_images, a):
    ms_ssim_loss = 1 - ms_ssim(attr_images, generated_images, data_range=1, size_average=True)
    l1_loss_value = l1_criterion(attr_images, generated_images)
    return a * ms_ssim_loss + (1 - a) * l1_loss_value


def id_loss(encoded_input_image, encoded_generated_image):
    return torch.norm(encoded_input_image - encoded_generated_image, p=1) / encoded_input_image.size()[0]


def landmark_loss(input_attr_lnd, output_lnd):
    loss = torch.norm(input_attr_lnd - output_lnd, p=2) / input_attr_lnd.size()[0]
    return loss
