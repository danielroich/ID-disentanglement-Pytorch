from Losses.AdversarialLoss import calc_Dw_loss
import torch

from Losses.NonAdversarialLoss import id_loss, landmark_loss, rec_loss


def train_discriminator(optimizer, real_w, generated_w, discriminator, ws, R1Param):
    optimizer.zero_grad()

    # 1.1 Train on Real Data
    prediction_real = discriminator(real_w).view(-1)
    # Calculate error and backpropagate
    error_real = calc_Dw_loss(prediction_real, 1, "cuda", ws, R1Param, False)
    error_real.backward()

    generated_w = generated_w.clone().detach()
    # 1.2 Train on Fake Data
    prediction_fake = discriminator(generated_w).view(-1)
    # Calculate error and backpropagate
    error_fake = calc_Dw_loss(prediction_fake, 0, generated_w, R1Param, False)

    error_fake.backward()

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error and predictions for real and fake inputs
    # return error_real + error_fake, prediction_real, prediction_fake
    return error_real, prediction_real, error_fake, prediction_fake


def train_mapper(optimizer, generated_w, discriminator, R1Param):
    optimizer.zero_grad()
    prediction = discriminator(generated_w).view(-1)
    # Calculate error and backpropagate
    error = calc_Dw_loss(prediction, 1, generated_w, R1Param, False)
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error, prediction


def discriminator_train_step(optimizerD, optimizerMLP, ws, fake_data, print_results=True):
    error_real, prediction_real, error_fake, prediction_fake = train_discriminator(optimizerD, ws, fake_data)
    g_error, g_pred = train_mapper(optimizerMLP, fake_data)

    if print_results:
        prediction_fake = torch.mean(prediction_fake)
        prediction_real = torch.mean(prediction_real)
        g_pred = torch.mean(g_pred)
        print(
            f"\n error_real: {error_real}, error_fake: {error_fake} \n prediction_real: {prediction_real}, prediction_fake: {prediction_fake}")
        print(f"\n g_error: {g_error}, g_pred: {g_pred}")

        return error_real, error_fake, prediction_real, prediction_fake, g_error, g_pred


def id_and_attr_train_step(generator,optimizerMLP, fake_data, id_transform,
                           attr_transform, config, E_id, E_lnd, id_vec, attr_images,
                           are_the_same_images=True, print_results=True):
    optimizerMLP.zero_grad()

    generated_images, _ = generator(
        [fake_data], input_is_latent=True, return_latents=False
    )
    generated_images = (generated_images + 1) / 2

    id_generated_images = id_transform(generated_images)
    attr_generated_images = attr_transform(generated_images)

    pred_id_embedding = torch.squeeze(E_id(id_generated_images))
    id_loss_val = config['lambdaID'] * id_loss(id_vec, pred_id_embedding)

    _, generated_landmarks = E_lnd(attr_generated_images.cpu().numpy())
    _, real_landmarks = E_lnd(attr_images)
    landmark_loss_val = config['lambdaLND'] * landmark_loss(generated_landmarks, real_landmarks)

    # if idx % config['IdDiffersAttrTrainRatio'] != 0:
    if are_the_same_images:
        rec_loss_val = config['lambdaREC'] * rec_loss(attr_images, generated_images, config['a'])
    else:
        rec_loss_val = 0

    total_error = rec_loss_val + id_loss_val + landmark_loss_val

    total_error.backward()
    optimizerMLP.step()

    if print_results:
        print(f"id_loss_val: {id_loss_val}")
        print(f"landmark_loss: {landmark_loss}")
        print(f"rec_loss: {rec_loss_val}")

    return id_loss_val, landmark_loss_val, rec_loss_val
