import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        slope = 0.2
        self.model = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=slope),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=slope),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=slope),
            nn.Linear(64, 1)
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a= slope)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_data):
        return self.model(input_data)

