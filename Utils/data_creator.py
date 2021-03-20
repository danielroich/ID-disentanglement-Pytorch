from Models.StyleGan2.model import Generator
from pathlib import Path
from torchvision import utils
from tqdm import tqdm
import torch.utils.data
import numpy as np
from Configs import Global_Config

BASE_PATH = Global_Config.BASE_PATH
GENERATOR_WEIGHTS_PATH = BASE_PATH + '550000.pt'
IMAGE_DATA_DIR = BASE_PATH + 'fake/small_image/'
W_DATA_DIR = BASE_PATH + 'fake/small_w/'

generator = Generator(256, 512, 8).eval().cuda()
state_dict = torch.load(GENERATOR_WEIGHTS_PATH)
generator.load_state_dict(state_dict['g_ema'], strict=False)

NUMBER_OF_IMAGES = 70000

with torch.no_grad():
    mean_latent = generator.mean_latent(4096 * 100)

counter = 0
cur_dir = 0
num_of_images_in_single_loop = 1
latents_total = []
for i in tqdm(range(NUMBER_OF_IMAGES // num_of_images_in_single_loop)):
    with torch.no_grad():
        sample_z = torch.randn(num_of_images_in_single_loop, 512, device='cuda')
        sample, latents = generator(
            [sample_z], input_is_latent=False, return_latents=True, truncation=0.5, truncation_latent=mean_latent
        )

    for index in range(len(sample)):
        if (counter % 1000) == 0:
            Path(f"{W_DATA_DIR}{int(counter / 1000)}").mkdir(parents=True, exist_ok=True)
            Path(f"{IMAGE_DATA_DIR}{int(counter / 1000)}").mkdir(parents=True, exist_ok=True)
            cur_dir = int(counter / 1000)

        with open(f'{W_DATA_DIR}{cur_dir}/{counter}.npy', 'wb') as f:
            np.save(f, latents[index][0])

        utils.save_image(
            sample[index],
            f'{IMAGE_DATA_DIR}{cur_dir}/{counter}.png',
            nrow=1,
            normalize=True,
            range=(-1, 1)
        )

        counter += 1
