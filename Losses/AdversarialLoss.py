import torch
import torch.nn as nn
from torch import autograd
from Configs import Global_Config


def calc_Dw_loss(probs: torch.Tensor, label: int):
    labels = torch.full((probs.size(0),), label, dtype=torch.float, device=Global_Config.device)
    criterion = nn.BCELoss()

    adversarial_loss = criterion(probs, labels)

    return adversarial_loss


def R1_regulazation(r1_coefficient, probs, ws):
    return (r1_coefficient / 2) * compute_grad2(probs, ws).mean()


def compute_grad2(probs, w_input):
    batch_size = w_input.size(0)
    grad_dout = autograd.grad(
        outputs=probs.sum(), inputs=w_input,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg
