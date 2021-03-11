from Losses.AdversarialLoss import calc_Dw_loss, R1_regulazation
import torch
from Losses.vgg_preceptual_loss import VGGPerceptualLoss
from Losses.NonAdversarialLoss import id_loss, landmark_loss, rec_loss


class Trainer:

    def __init__(self, config,
                 discriminator_optimizer: torch.optim.Optimizer,
                 adversarial_mapper_optimizer: torch.optim.Optimizer,
                 non_adversarial_mapper_optimizer: torch.optim.Optimizer,
                 discriminator,
                 generator,
                 id_transform,
                 attr_transform,
                 landmark_transform,
                 id_encoder,
                 attr_encoder,
                 landmark_encoder,
                 is_grad=False):

        self.config = config
        self.discriminator_optimizer = discriminator_optimizer
        self.adversarial_mapper_optimizer = adversarial_mapper_optimizer
        self.non_adversarial_mapper_optimizer = non_adversarial_mapper_optimizer
        self.discriminator = discriminator
        self.discriminator = discriminator
        self.generator = generator
        self.id_transform = id_transform
        self.attr_transform = attr_transform
        self.landmark_transform = landmark_transform
        self.id_encoder = id_encoder
        self.attr_encoder = attr_encoder
        self.landmark_encoder = landmark_encoder
        self.vgg_loss = VGGPerceptualLoss(is_grad=is_grad).cuda()

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

    def non_adversarial_train_step_with_vgg(self, id_vec, attr_images, fake_data, is_grad=False):
        self.id_encoder.zero_grad()
        self.landmark_encoder.zero_grad()
        self.generator.zero_grad()
        self.vgg_loss.zero_grad()

        rec_loss_val = torch.tensor(0)
        id_loss_val = torch.tensor(0)
        landmark_loss_val = torch.tensor(0)

        generated_images, _ = self.generator(
            [fake_data], input_is_latent=True, return_latents=False
        )
        generated_images = (generated_images + 1) / 2

        if self.config['use_id']:
            id_generated_images = self.id_transform(generated_images)
            pred_id_embedding = torch.squeeze(self.id_encoder(id_generated_images))
            id_loss_val = self.config['lambdaID'] * id_loss(id_vec, pred_id_embedding)

        if self.config['use_landmark']:
            landmark_attr_images = self.landmark_transform(attr_images)
            landmark_generated_images = self.landmark_transform(generated_images)
            generated_landmarks, generated_landmarks_nojawline = self.landmark_encoder(landmark_generated_images)
            real_landmarks, real_landmarks_nojawline = self.landmark_encoder(landmark_attr_images)
            landmark_loss_val = landmark_loss(generated_landmarks, real_landmarks) * self.config['lambdaLND']

        if self.config['use_reconstruction']:
            rec_loss_val = self.config['lambdaREC'] * rec_loss(attr_images, generated_images, self.config['a'])

        vgg_loss_val = self.config['lambdaVGG'] * self.vgg_loss(generated_images, attr_images, feature_layers=[2],
                                                                style_layers=[0, 1, 2, 3])

        total_error = rec_loss_val + id_loss_val + landmark_loss_val + vgg_loss_val

        self.non_adversarial_mapper_optimizer.zero_grad()
        total_error.backward()
        self.non_adversarial_mapper_optimizer.step()

        return id_loss_val, rec_loss_val, landmark_loss_val, vgg_loss_val, total_error

    def non_adversarial_train_step(self, id_vec, attr_images, fake_data):
        self.id_encoder.zero_grad()
        self.landmark_encoder.zero_grad()
        self.generator.zero_grad()
        self.vgg_loss.zero_grad()
        self.attr_encoder.zero_grad()

        rec_loss_val = torch.tensor(0)
        id_loss_val = torch.tensor(0)
        landmark_loss_val = torch.tensor(0)

        generated_images, _ = self.generator(
            [fake_data], input_is_latent=True, return_latents=False
        )
        generated_images = (generated_images + 1) / 2

        if self.config['use_id']:
            id_generated_images = self.id_transform(generated_images)
            pred_id_embedding = torch.squeeze(self.id_encoder(id_generated_images))
            id_loss_val = self.config['lambdaID'] * id_loss(id_vec, pred_id_embedding)

        if self.config['use_landmark']:
            landmark_attr_images = self.landmark_transform(attr_images)
            landmark_generated_images = self.landmark_transform(generated_images)
            generated_landmarks, generated_landmarks_nojawline = self.landmark_encoder(landmark_generated_images)
            real_landmarks, real_landmarks_nojawline = self.landmark_encoder(landmark_attr_images)
            landmark_loss_val = landmark_loss(generated_landmarks, real_landmarks) * self.config['lambdaLND']

        if self.config['use_reconstruction']:
            rec_loss_val = self.config['lambdaREC'] * rec_loss(attr_images, generated_images, self.config['a'])

        total_error = rec_loss_val + id_loss_val + landmark_loss_val

        self.non_adversarial_mapper_optimizer.zero_grad()
        total_error.backward()
        self.non_adversarial_mapper_optimizer.step()

        return id_loss_val, rec_loss_val, landmark_loss_val, total_error
