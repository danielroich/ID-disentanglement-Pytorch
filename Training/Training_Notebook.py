#!/usr/bin/env python
# coding: utf-8

# # clone + import + drive + paths
# 

# In[1]:


import os
import Global_Config


# In[2]:


config = {
    'beta1' : 0.9,
    'beta2' : 0.999,
    'adverserial_D' : 2e-5,
    'adverserial_M' : 5e-6,
    'non_adverserial_lr' : 5e-5,
    'lrAttr' : 0.0001,
    'IdDiffersAttrTrainRatio' : 3, # 1/3
    'batchSize' : 8,
    'R1Param' : 10,
    'lambdaID' : 1,
    'lambdaLND' : 1,
    'lambdaREC' : 2,
    'lambdaVGG': 0.0004,
    'a' : 0.84,
    'use_reconstruction' : True,
    'use_id' : False,
    'use_landmark' : True,
    'use_adverserial' : False,
    'train_precentege' : 0.95,
    'epochs' : 40
}
GENERATOR_IMAGE_SIZE = 256


# In[3]:


def get_base_path(run_in_colab):
    if Global_Config.run_in_slurm:
        return '/home/joberant/nlp_fall_2021/danielroich/disantalgement/Data/'
    if run_in_colab:
        return '/content/drive/MyDrive/CNN-project-weights/'
    return '/disk2/danielroich/yotam_disentanglement/Data/'


# In[4]:


BASE_PATH = get_base_path(Global_Config.run_in_colab)

MOBILE_FACE_NET_WEIGHTS_PATH = BASE_PATH + 'mobilefacenet_model_best.pth.tar'
GENERATOR_WEIGHTS_PATH = BASE_PATH + '550000.pt'
E_ID_WEIGHTS_PATH = BASE_PATH + 'resnet50_scratch_dag.pth'
E_ID_NEW__WEIGHTS_PATH = BASE_PATH + 'resnet50_scratch_weight.pkl'
DLIB_WEIGHT_PATH = BASE_PATH + 'mmod_human_face_detector.dat'
IMAGE_DATA_DIR = BASE_PATH + 'fake/small_image/'
W_DATA_DIR = BASE_PATH + 'fake/small_w/'
MODELS_DIR = BASE_PATH + 'Models/'


# In[5]:


def prepeare_env_for_local_use():
    CUDA_VISIBLE_DEVICES = '4'
    os.chdir('..')
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']= CUDA_VISIBLE_DEVICES


# In[6]:


def prepeare_env_for_colab():
    from google.colab import drive
    drive.mount('/content/drive')
    os.chdir('/content')
    get_ipython().system('pip install pytorch-msssim')
    get_ipython().system("git clone https://github.com/danielroich/Face-Identity-Disentanglement-via-StyleGan2.git 'project'")
    CODE_DIR = 'project'
    os.chdir(f'./{CODE_DIR}')


# In[7]:


if Global_Config.run_in_colab:
    prepeare_env_for_colab()
elif Global_Config.run_in_slurm:
    os.chdir('..')
else:
    prepeare_env_for_local_use()


# In[8]:


import wandb
from Training.trainer import Trainer
from torch.utils.data import DataLoader, random_split
from Models.Encoders.Landmark_Encoder import Landmark_Encoder
from Models.Encoders.ID_Encoder import ID_Encoder
from Models.Encoders.Inception import Inception
from Models.Discrimanator import Discriminator
from Models.LatentMapper import LatentMapper
from Models.StyleGan2.model import Generator
from Utils.data_utils import get_w_image, plot_single_w_image, Image_W_Dataset
import time
import torch
import torch.utils.data
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


# # network + data
# 

# In[9]:


id_encoder = ID_Encoder()
attr_encoder = Inception()
discriminator = Discriminator()
mlp = LatentMapper()
landmark_encoder = Landmark_Encoder.Encoder_Landmarks(MOBILE_FACE_NET_WEIGHTS_PATH)
generator = Generator(GENERATOR_IMAGE_SIZE, 512, 8)


# In[10]:

state_dict = torch.load(GENERATOR_WEIGHTS_PATH)
generator.load_state_dict(state_dict['g_ema'], strict=False)


# In[12]:


id_encoder = id_encoder.to(Global_Config.device)
attr_encoder = attr_encoder.to(Global_Config.device)
discriminator = discriminator.to(Global_Config.device)
mlp = mlp.to(Global_Config.device)
generator = generator.to(Global_Config.device)
landmark_encoder = landmark_encoder.to(Global_Config.device)


# In[13]:


id_encoder = id_encoder.eval()
attr_encoder = attr_encoder.train()
discriminator = discriminator.train()
generator = generator.eval()
mlp = mlp.train()
landmark_encoder = landmark_encoder.eval()


# In[14]:


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


# In[15]:


toggle_grad(id_encoder, True)
toggle_grad(attr_encoder, True)
toggle_grad(generator, True)
toggle_grad(mlp, True)
toggle_grad(landmark_encoder, True)


# In[17]:


w_image_dataset = Image_W_Dataset(W_DATA_DIR, IMAGE_DATA_DIR)


# In[18]:


# train_size = int(config['train_precentege'] * len(w_image_dataset))
train_size = 65000
test_size = len(w_image_dataset) - train_size
train_data, test_data = random_split(w_image_dataset, [train_size, test_size])


# In[19]:


train_loader = DataLoader(dataset=train_data, batch_size=config['batchSize'], shuffle=False)
test_loader = DataLoader(dataset=test_data, batch_size=config['batchSize'], shuffle=False)


# # Training

# In[20]:


optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config['adverserial_D'], betas=(config['beta1'], config['beta2']))
optimizer_adv_M = torch.optim.Adam(mlp.parameters(), lr=config['adverserial_M'], betas=(config['beta1'], config['beta2']))
optimizer_non_adv_M = torch.optim.Adam(list(mlp.parameters()) + list(attr_encoder.parameters()), lr=config['non_adverserial_lr'], betas=(config['beta1'], config['beta2']))


# In[21]:


trainer = Trainer(config, optimizer_D, optimizer_adv_M,optimizer_non_adv_M,discriminator,generator,
                  id_encoder, attr_encoder, landmark_encoder, True)


# In[22]:


run = wandb.init(project="yotam_disantalgement", reinit=True, config = config)


# In[23]:


def get_concat_vec(id_images, attr_images, id_encoder, attr_encoder):
    with torch.no_grad():
        id_vec = torch.squeeze(id_encoder(id_images))
        non_cycled_attr_vec = torch.squeeze(attr_encoder(attr_images))
        non_cycled_test_vec = torch.cat((id_vec, non_cycled_attr_vec), dim=1)
        return non_cycled_test_vec


# In[25]:


ws, images  = next(iter((test_loader)))

test_id_images = images.to(Global_Config.device).clone()
test_attr_images = images.to(Global_Config.device).clone()
test_ws = ws.to(Global_Config.device)

with torch.no_grad():
    image1 = get_w_image(test_ws[0], generator)
    image12 = get_w_image(test_ws[1], generator)

    if Global_Config.run_in_notebook:
        print('ID image1:')
        plot_single_w_image(test_ws[0], generator)
        print('ID image2:')
        plot_single_w_image(test_ws[1], generator)

    wandb.log({"Test_ID_Image1": [wandb.Image(image1 * 255, caption="ID image1")]}, step=0)
    wandb.log({"Test_ID_Image2": [wandb.Image(image12 * 255, caption="ID image2")]}, step=0)


# ## Global Training

# In[26]:


def mean(tensors_list):
  if len(tensors_list) == 0:
    return 0
  return sum(tensors_list) / len(tensors_list)


# In[27]:


D_error_real_train = []
D_error_fake_train = []
D_prediction_real_train = []
D_prediction_fake_train = []
G_error_train = []
G_pred_train = []
id_loss_train = []
rec_loss_train = []
landmark_loss_train = []
vgg_loss_train = []
total_error_train = []


# In[ ]:


use_descrimantive_loss = config['use_adverserial']
total_error = 0
with tqdm(total=config['epochs'] * len(train_loader)) as pbar:
    for epoch in range(config['epochs']):
        for idx, data in enumerate(train_loader):
            ws, images = data

            id_images = images.detach().clone().to(Global_Config.device)
            attr_images = images.detach().clone().to(Global_Config.device)
            ws = ws.to(Global_Config.device)

            # if idx % config['IdDiffersAttrTrainRatio'] == 0:
            #   attr_images = cycle_images_to_create_diff_order(attr_images)

            try:
                with torch.no_grad():
                    id_vec = torch.squeeze(id_encoder(id_images))
                    real_landmarks, real_landmarks_nojawline = landmark_encoder(attr_images)
            except Exception as e:
                print(e)

            attr_vec = torch.squeeze(attr_encoder(attr_images))
            try:
                encoded_vec = torch.cat((id_vec, attr_vec), dim=1)
            except Exception as e:
                print(e)

            fake_data = mlp(encoded_vec)

            if use_descrimantive_loss and idx % 2 == 0:
                error_real, error_fake, prediction_real, prediction_fake, g_error, g_pred = trainer.adversarial_train_step(
                    ws, fake_data)
                D_error_real_train.append(error_real.cpu().detach())
                D_error_fake_train.append(error_fake.cpu().detach())
                D_prediction_real_train.append(prediction_real.cpu().detach())
                D_prediction_fake_train.append(prediction_fake.cpu().detach())
                G_error_train.append(g_error.cpu().detach())
                G_pred_train.append(g_pred.cpu().detach())

            else:
                # use_rec = (idx % config['IdDiffersAttrTrainRatio'] != 0)
                use_rec = True
                id_loss_val, rec_loss_val, landmark_loss_val, vgg_loss_val, total_error = trainer.non_adversarial_train_step(
                    id_vec, attr_images, fake_data, real_landmarks)
                id_loss_train.append(id_loss_val)
                rec_loss_train.append(rec_loss_val)
                landmark_loss_train.append(landmark_loss_val)
                vgg_loss_train.append(vgg_loss_val)
                total_error_train.append(total_error)

            pbar.update(1)
            if idx % 30 == 0 and idx != 0:
                with torch.no_grad():
                    if Global_Config.run_in_notebook:
                        plot_single_w_image(mlp(
                            get_concat_vec(test_id_images, test_attr_images, id_encoder, attr_encoder))[0], generator)

                    wandb.log(
                        {'D_error_real_train': mean(D_error_real_train), 'D_error_fake_train': mean(D_error_fake_train),
                         'D_prediction_real_train': mean(D_prediction_real_train),
                         'D_prediction_fake_train': mean(D_prediction_fake_train),
                         'G_error_train': mean(G_error_train), 'G_pred_train': mean(G_pred_train),
                         'id_loss_train': mean(id_loss_train), 'rec_loss_train': mean(rec_loss_train),
                         'landmark_loss_train': mean(landmark_loss_train), 'total_error_train': mean(total_error_train),
                         'vgg_loss_train': mean(vgg_loss_train)})
                    D_error_real_train = []
                    D_error_fake_train = []
                    D_prediction_real_train = []
                    D_prediction_fake_train = []
                    G_error_train = []
                    G_pred_train = []
                    id_loss_train = []
                    rec_loss_train = []
                    landmark_loss_train = []
                    vgg_loss_train = []
                    total_error_train = []

            if idx % 30 == 0 and idx != 0:
                with torch.no_grad():
                    concat_vec = get_concat_vec(test_id_images, test_attr_images, id_encoder, attr_encoder)
                    id_generated_image = get_w_image(mlp(concat_vec)[0], generator)
                    id_generated_image2 = get_w_image(mlp(concat_vec)[1], generator)
                    wandb.log(
                        {"ID_Train_Images1": [wandb.Image(id_generated_image * 255, caption=f"generated_image{idx}")]})
                    wandb.log({"ID_Train_Images2": [
                        wandb.Image(id_generated_image2 * 255, caption=f"generated_image{idx}_2")]})
                    # wandb.log({"Cycle_Train_Images": [wandb.Image(id_and_attr_generated_image* 255, caption=f"generated_image{idx}")]})

            if idx % 600 == 0 and idx != 0:
                torch.save(mlp, f'{MODELS_DIR}maper_{idx}_{time.time()}_{int(total_error)}.pt')
                torch.save(discriminator, f'{MODELS_DIR}discriminator_{idx}_{time.time()}_{int(total_error)}.pt')


# In[ ]:




