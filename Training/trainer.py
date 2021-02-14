from Losses.AdversarialLoss import calc_Dw_loss
import torch

from Losses.NonAdversarialLoss import id_loss, landmark_loss, rec_loss


class Trainer:

    def __init__(self, config,
                 discriminator_optimizer: torch.optim.Optimizer,
                 mapper_optimizer: torch.optim.Optimizer,
                 discriminator,
                 generator,
                 id_transform,
                 attr_transform,
                 landmark_transform,
                 id_encoder,
                 attr_encoder,
                 landmark_encoder):

        self.config = config
        self.optimizer_D = discriminator_optimizer
        self.optimizer_M = mapper_optimizer
        self.discriminator = discriminator
        self.discriminator = discriminator
        self.generator = generator
        self.id_transform = id_transform
        self.attr_transform = attr_transform
        self.landmark_transform = landmark_transform
        self.id_encoder = id_encoder
        self.attr_encoder = attr_encoder
        self.landmark_encoder = landmark_encoder

    def train_discriminator(self, real_w, generated_w):
        self.optimizer_D.zero_grad()

        # 1.1 Train on Real Data
        prediction_real = self.discriminator(real_w).view(-1)
        # Calculate error and backpropagate
        error_real = calc_Dw_loss(prediction_real, 1, real_w, self.config['R1Param'], False)
        error_real.backward()

        cloned_generated_w = generated_w.clone().detach()
        # 1.2 Train on Fake Data
        prediction_fake = self.discriminator(cloned_generated_w).view(-1)
        # Calculate error and backpropagate
        error_fake = calc_Dw_loss(prediction_fake, 0, cloned_generated_w, self.config['R1Param'], False)

        error_fake.backward()

        # 1.3 Update weights with gradients
        self.optimizer_D.step()

        # Return error and predictions for real and fake inputs
        # return error_real + error_fake, prediction_real, prediction_fake
        return error_real, prediction_real, error_fake, prediction_fake

    def train_mapper(self, generated_w):
        self.optimizer_M.zero_grad()
        prediction = self.discriminator(generated_w).view(-1)
        # Calculate error and backpropagate
        error = calc_Dw_loss(prediction, 1, generated_w, self.config['R1Param'], False)
        error.backward()
        # Update weights with gradients
        self.optimizer_M.step()
        # Return error
        return error, prediction

    def adversarial_train_step(self, real_w, fake_data, print_results=True):
        error_real, prediction_real, error_fake, prediction_fake = self.train_discriminator(real_w, fake_data)
        g_error, g_pred = self.train_mapper(fake_data)

        if print_results:
            prediction_fake = torch.mean(prediction_fake)
            prediction_real = torch.mean(prediction_real)
            g_pred = torch.mean(g_pred)
            print(
                f"\n error_real: {error_real}, error_fake: {error_fake} \n prediction_real: {prediction_real}, prediction_fake: {prediction_fake}")
            print(f"\n g_error: {g_error}, g_pred: {g_pred}")

            return error_real, error_fake, prediction_real, prediction_fake, g_error, g_pred

    def non_adversarial_train_step(self, fake_data,
                                   original_id_vec, original_attr_images,
                                   are_the_same_images=True, print_results=True):

        self.optimizer_M.zero_grad()

        generated_images, _ = self.generator(
            [fake_data], input_is_latent=True, return_latents=False
        )
        generated_images = (generated_images + 1) / 2

        id_generated_images = self.id_transform(generated_images)

        pred_id_embedding = torch.squeeze(self.id_encoder(id_generated_images))
        id_loss_val = self.config['lambdaID'] * id_loss(original_id_vec, pred_id_embedding)

        landmark_attr_images = self.landmark_transform(original_attr_images).permute(0, 2, 3, 1) \
                                    .cpu().numpy() * 255
        landmark_generated_images = self.landmark_transform(generated_images.detach()) \
                                         .permute(0, 2, 3, 1).cpu().numpy() * 255

        try:
            _, generated_landmarks = self.landmark_encoder(landmark_generated_images)
            _, real_landmarks = self.landmark_encoder(landmark_attr_images)
            landmark_loss_val = self.config['lambdaLND'] * landmark_loss(generated_landmarks, real_landmarks)

        except Exception as e:
            landmark_loss_val = 0
            if print_results:
                print(str(e))

        if are_the_same_images:
            rec_loss_val = self.config['lambdaREC'] * rec_loss(original_attr_images,
                                                               generated_images, self.config['a'])
        else:
            rec_loss_val = 0

        total_error = rec_loss_val + id_loss_val + landmark_loss_val

        total_error.backward()
        self.optimizer_M.step()

        if print_results:
            print(f"id_loss_val: {id_loss_val}")
            print(f"landmark_loss: {landmark_loss_val}")
            print(f"rec_loss: {rec_loss_val}")

        return id_loss_val, rec_loss_val
