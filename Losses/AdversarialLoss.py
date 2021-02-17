import torch
import torch.nn as nn
from torch.autograd import grad


def calc_Dw_loss(probs: torch.Tensor, label: int, inputs: torch.Tensor, r1_gamma: int, add_regulrization: bool,
                 device: torch.device):

    labels = torch.full((probs.size(0),), label, dtype=torch.float, device=device)
    grad_penalty = 0

    # Add R1 regularization only if the label is real
    if add_regulrization:
        # grad_real = grad(outputs=x_outputs.sum(), inputs=x, create_graph=True)[0]
        grad_real = grad(outputs=probs, inputs=inputs, grad_outputs=torch.ones_like(probs), create_graph=True)[0]
        grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
        grad_penalty = 0.5 * r1_gamma * grad_penalty

    criterion = nn.BCELoss()

    adversarial_loss = criterion(probs, labels)

    return adversarial_loss + grad_penalty
