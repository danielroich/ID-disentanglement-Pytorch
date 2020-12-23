from torch import nn


def L1_consistensy_loss(encoded_input_image, encoded_generated_image):
    return nn.L1Loss(encoded_input_image, encoded_generated_image)
