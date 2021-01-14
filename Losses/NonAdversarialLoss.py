from pytorch_msssim import ms_ssim
import numpy as np
import torch

def rec_loss(attr_images, generated_images, a):
  ms_ssim_loss = 1 - ms_ssim( attr_images, generated_images, data_range=1, size_average=True )
  l1_loss = torch.nn.L1Loss(reduction = 'mean')
  l1_loss_value = l1_loss(attr_images, generated_images)
  return a * ms_ssim_loss + (1-a) * l1_loss_value


def id_loss(encoded_input_image, encoded_generated_image):
    loss = torch.nn.L1Loss(reduction = 'mean')
    return loss(encoded_input_image, encoded_generated_image)

def landmark_loss(input_attr_lnd, output_lnd):
    loss = torch.norm(input_attr_lnd - output_lnd, p=2)
    return loss