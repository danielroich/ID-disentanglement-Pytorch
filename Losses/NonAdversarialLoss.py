from pytorch_msssim import MS_SSIM
import numpy as np
import torch

def reconstruction_loss(Iattr, Iout,Iid, a):

    img_attr = torch.from_numpy(np.rollaxis(Iattr, 2)).float().unsqueeze(0)/255.0
    if torch.cuda.is_available():
      img_attr = img_attr.cuda()
    img_out = torch.from_numpy(np.rollaxis(Iout, 2)).float().unsqueeze(0)/255.0
    if torch.cuda.is_available():
      img_out = img_out.cuda()
    # img_attr, img_out: (N,3,H,W) a batch of non-negative RGB images (0~255)
    ms_ssim_val = ms_ssim( img_attr, img_out, data_range=255, size_average=False )
    norm1=np.linalg.norm(np.asarray(Iattr)-np.asarray(Iout))
    return a*(1-ms_ssim_val)+(1-a)*norm1


def id_loss(encoded_input_image, encoded_generated_image):
    loss = torch.nn.L1Loss()
    return loss(encoded_input_image, encoded_generated_image)

def landmark_loss(input_attr_lnd, output_lnd):
    loss = torch.norm(input_attr_lnd - output_lnd, p=2)
    return loss