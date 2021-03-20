import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from Configs import Global_Config

to_tensor_transform = transforms.ToTensor()


def plot_single_w_image(w, generator):
    w = w.unsqueeze(0).to(Global_Config.device)
    sample, latents = generator(
        [w], input_is_latent=True, return_latents=True
    )
    new_image = sample.cpu().detach().numpy().transpose(0, 2, 3, 1)[0]
    new_image = (new_image + 1) / 2
    plt.axis('off')
    plt.imshow(new_image)
    plt.show()


def get_w_image(w, generator):
    w = w.unsqueeze(0).to(Global_Config.device)
    sample, latents = generator(
        [w], input_is_latent=True, return_latents=True
    )
    new_image = sample.cpu().detach().numpy().transpose(0, 2, 3, 1)[0]
    new_image = (new_image + 1) / 2

    return new_image


def get_data_by_index(idx, root_dir, postfix):
    if torch.is_tensor(idx):
        idx = idx.tolist()

    dir_idx = idx // 1000

    path = os.path.join(root_dir, str(dir_idx), str(idx) + postfix)
    if postfix == ".npy":
        data = torch.tensor(np.load(path))

    elif postfix == ".png":
        data = to_tensor_transform(Image.open(path))

    else:
        return None

    return data


class Image_W_Dataset(Dataset):
    def __init__(self, w_dir, image_dir):
        self.w_dir = w_dir
        self.image_dir = image_dir

    def __len__(self):
        num_of_files = 0
        for base, dirs, files in os.walk(self.w_dir):
            num_of_files += len(files)
        return num_of_files

    def __getitem__(self, idx):
        w = get_data_by_index(idx, self.w_dir, ".npy")
        image = get_data_by_index(idx, self.image_dir, ".png")
        return w, image


def cycle_images_to_create_diff_order(images):
    batch_size = len(images)
    different_images = torch.empty_like(images, device=Global_Config.device)
    different_images[0] = images[batch_size - 1]
    different_images[1:] = images[:batch_size - 1]
    return different_images
