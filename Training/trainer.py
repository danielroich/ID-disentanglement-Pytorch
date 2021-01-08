from Losses.AdversarialLoss import calc_Dw_loss
from Models.Encoders.ID_Encoder import resnet50_scratch_dag
from Models.Encoders.Attribute_Encoder import Encoder_Attribute
from Models.Discrimanator import Discriminator
from Models.LatentMapper import MLPModel
import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np


def get_data(mean, std, image_size, data_dir):
    full_dataset = dset.ImageFolder(root=data_dir,
                                    transform=transforms.Compose([
                                        transforms.Resize(image_size),
                                        transforms.ToTensor(),
                                    ]))

    return full_dataset


def make_loaders(dataset, batch_size):
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size)

    return train_loader

### batch_size <= len(real_w)
def check_overfit_for_single_batch(batch_size=8):

    ####### Create Models #######
    id_encoder = resnet50_scratch_dag(weights_path=r'..\Weights\resnet50_scratch_dag.pth')

    # The model should be kept frozen during training time
    id_encoder.eval()

    attr_encoder = Encoder_Attribute()
    discriminator = Discriminator(512, 512)
    mlp = MLPModel(4096, n_hid=2048)

    ####### Create data #######
    data_folder = get_data(id_encoder.meta['mean'], id_encoder.meta['std'], id_encoder.meta['imageSize'][1],
                           r'..\Datasets\fake_256\images')
    train_loader = make_loaders(data_folder, batch_size)
    images, _ = next(iter(train_loader))

    ####### Forward Pass #######
    _, id_vec = id_encoder(images)
    #_, attr_vec = attr_encoder(images)
    id_vec = torch.squeeze(id_vec)
    attr_vec = torch.squeeze(id_vec)
    encoded_vec = np.concatenate((id_vec,attr_vec))
    generated_w_vec = mlp(encoded_vec)

    ####### Discriminator back pass #######
    fake_prob = discriminator(generated_w_vec).view(-1)
    errD_fake = calc_Dw_loss(fake_prob, 0, "cpu", generated_w_vec, 0.01)
    real_w = np.load(r'..\Datasets\fake_256\ws\00000\00001.npy')
    real_prob = discriminator(real_w).view(-1)
    errD_real = calc_Dw_loss(real_prob, 0, "cpu", generated_w_vec, 0.01)
    real_w_batch = real_w[:batch_size]


if __name__ == "__main__":
    check_overfit_for_single_batch()
