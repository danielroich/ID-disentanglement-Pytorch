from pytorch_msssim import MS_SSIM
import numpy as np
import torch

def is_similar(image1, image2):
    return image1.shape == image2.shape and not(np.bitwise_xor(image1,image2).any())


def Lrec(Iattr, Iout,Iid, a):
  if is_similar(Iattr, Iid):
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
  else:
    return 0