import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

to_tensor_transform = transforms.ToTensor()

def plot_single_w_image(w, generator):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    w = w.unsqueeze(0).to(device)
    sample, latents = generator(
        [w], input_is_latent=True, return_latents=True
    )
    new_image = sample.cpu().detach().numpy().transpose(0, 2, 3, 1)[0]
    new_image = (new_image + 1) / 2
    plt.axis('off')
    plt.imshow(new_image)
    plt.show()


def get_w_image(w, generator):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    w = w.unsqueeze(0).to(device)
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


class WDataSet(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the w's.
        """
        self.root_dir = root_dir

    def __len__(self):
        num_of_files = 0
        for base, dirs, files in os.walk(self.root_dir):
            num_of_files += len(files)
        return num_of_files

    def __getitem__(self, idx):
        return get_w_by_index(idx, self.root_dir)


class ConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


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
        return w,image


def make_concat_loaders(batch_size, datasets):
    full_dataset = ConcatDataset(datasets)

    train_loader = torch.utils.data.DataLoader(dataset=full_dataset,
                                               batch_size=batch_size, shuffle=True)

    return train_loader


def cycle_images_to_create_diff_order(images):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = len(images)
    different_images = torch.empty_like(images, device=device)
    different_images[0] = images[batch_size - 1]
    different_images[1:] = images[:batch_size - 1]
    return different_images
