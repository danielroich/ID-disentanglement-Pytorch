import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from torch.utils.data import Dataset


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

    return new_image


def get_w_by_index(idx, root_dir):
    if torch.is_tensor(idx):
        idx = idx.tolist()

    dir_idx = idx // 1000

    w_path = os.path.join(root_dir, str(dir_idx), str(idx) + ".npy")
    w = np.load(w_path)

    return torch.tensor(w)


class WDataSet(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the w's.
        """
        self.root_dir = root_dir

    def __len__(self):
        path, dirs, files = next(os.walk(self.root_dir))
        return len(files)
        # ## TODO: Change
        # return 6999

    def __getitem__(self, idx):
        return get_w_by_index(idx, self.root_dir)


class ConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


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
