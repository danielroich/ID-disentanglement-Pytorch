from torchvision import transforms

from Losses.AdversarialLoss import calc_Dw_loss, R1_regulazation
import torch
from Losses.NonAdversarialLoss import id_loss, landmark_loss, rec_loss, VGGLoss, l2_loss
import Global_Config
from Losses import id_loss
from Losses.lpips.lpips import LPIPS


class Trainer:

    def __init__(self, config,
                 discriminator_optimizer: torch.optim.Optimizer,
                 adversarial_mapper_optimizer: torch.optim.Optimizer,
                 non_adversarial_mapper_optimizer: torch.optim.Optimizer,
                 discriminator,
                 generator,
                 id_encoder,
                 attr_encoder,
                 landmark_encoder,
                 id_loss_pth):

        self.config = config
        self.discriminator_optimizer = discriminator_optimizer
        self.adversarial_mapper_optimizer = adversarial_mapper_optimizer
        self.non_adversarial_mapper_optimizer = non_adversarial_mapper_optimizer
        self.discriminator = discriminator
        self.discriminator = discriminator
        self.generator = generator
        self.id_encoder = id_encoder
        self.attr_encoder = attr_encoder
        self.landmark_encoder = landmark_encoder
        self.id_loss = id_loss.IDLoss(id_loss_pth).to(Global_Config.device).eval()
        self.lpips_loss = LPIPS(net_type='vgg').to(Global_Config.device).eval()
        self.vgg_loss = VGGLoss().to(Global_Config.device)
        self.vgg_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])

    def train_discriminator(self, real_w, generated_w):
        self.discriminator_optimizer.zero_grad()
        real_w.requires_grad_()

        prediction_real = self.discriminator(real_w).view(-1)
        error_real = calc_Dw_loss(prediction_real, 1)
        error_real.backward(retain_graph=True)

        r1_error = R1_regulazation(self.config['R1Param'], prediction_real, real_w)
        r1_error.backward()

        cloned_generated_w = generated_w.clone().detach()
        prediction_fake = self.discriminator(cloned_generated_w).view(-1)
        error_fake = calc_Dw_loss(prediction_fake, 0)
        error_fake.backward()

        self.discriminator_optimizer.step()

        return error_real, prediction_real, error_fake, prediction_fake

    def train_mapper(self, generated_w):

        self.adversarial_mapper_optimizer.zero_grad()
        prediction = self.discriminator(generated_w).view(-1)
        discriminative_loss = calc_Dw_loss(prediction, 1)
        discriminative_loss.backward()
        self.adversarial_mapper_optimizer.step()

        return discriminative_loss, prediction

    def adversarial_train_step(self, real_w, fake_data):
        error_real, prediction_real, error_fake, prediction_fake = self.train_discriminator(real_w, fake_data)
        g_error, g_pred = self.train_mapper(fake_data)

        return error_real, error_fake, torch.mean(prediction_real), torch.mean(prediction_fake), g_error, torch.mean(
            g_pred)

    def non_adversarial_train_step(self, id_images, attr_images, fake_data, real_landmarks, use_rec_extra_term):
        self.id_encoder.zero_grad()
        self.landmark_encoder.zero_grad()
        self.generator.zero_grad()
        self.vgg_loss.zero_grad()

        total_loss = torch.tensor(0, dtype=torch.float, device=Global_Config.device)
        rec_loss_val = torch.tensor(0, dtype=torch.float, device=Global_Config.device)
        id_loss_val = torch.tensor(0, dtype=torch.float, device=Global_Config.device)
        landmark_loss_val = torch.tensor(0, dtype=torch.float, device=Global_Config.device)
        vgg_loss_val = torch.tensor(0, dtype=torch.float, device=Global_Config.device)
        l2_loss_val = torch.tensor(0, dtype=torch.float, device=Global_Config.device)

        generated_images, _ = self.generator(
            [fake_data], input_is_latent=True, return_latents=False
        )
        ## TODO: Check for each net the image scale
        ## TODO: Pip install lpips
        normalized_generated_images = (generated_images + 1) / 2

        ## -1 to 1
        if self.config['use_id']:
            # pred_id_embedding = torch.squeeze(self.id_encoder(generated_images))
            # id_loss_val = self.config['lambdaID'] * id_loss(real_id_vec, pred_id_embedding)
            id_loss_val = self.config['lambdaID'] * self.id_loss(normalized_generated_images, id_images)
            total_loss += id_loss_val

        if self.config['use_landmark']:
            generated_landmarks, generated_landmarks_nojawline = self.landmark_encoder(normalized_generated_images)
            landmark_loss_val = landmark_loss(generated_landmarks, real_landmarks) * self.config['lambdaLND']
            total_loss += landmark_loss_val

        ## 0 to 1
        if use_rec_extra_term and self.config['use_reconstruction']:
            rec_loss_val = self.config['lambdaREC'] * rec_loss(attr_images, normalized_generated_images,
                                                               self.config['a'])
            total_loss += rec_loss_val

        if self.config['use_l2'] > 0:
            l2_loss_val = self.config['lambdaL2'] * l2_loss(attr_images, normalized_generated_images)
            total_loss += l2_loss_val

        # 0 to 1 and then normalize according to official site
        if use_rec_extra_term and (not self.config['use_adverserial']):
            vgg_loss_val = self.config['lambdaVGG'] * self.lpips_loss(normalized_generated_images, attr_images)
            # vgg_loss_val = self.config['lambdaVGG'] * self.vgg_loss(self.vgg_normalize(attr_images),
            #                                                         self.vgg_normalize(normalized_generated_images))

        self.non_adversarial_mapper_optimizer.zero_grad()
        total_loss.backward()
        self.non_adversarial_mapper_optimizer.step()

        return id_loss_val, rec_loss_val, landmark_loss_val, vgg_loss_val, l2_loss_val, total_loss
