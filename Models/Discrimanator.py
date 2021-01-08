import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, num_features, n_hid = 256):
        super(Discriminator, self).__init__()
        slope = 0.2
        self.model = nn.Sequential(
            nn.Linear(num_features, n_hid),
            nn.LeakyReLU(negative_slope=slope),
            #nn.BatchNorm1d(n_hid, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Linear(n_hid, n_hid // 2),
            nn.LeakyReLU(negative_slope=slope),
            # nn.BatchNorm1d(n_hid // 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Linear(n_hid // 2, n_hid // 4),
            nn.LeakyReLU(negative_slope=slope),
            # nn.BatchNorm1d(n_hid // 4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Linear(n_hid // 4, 1),
            #nn.Sigmoid()
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_data):
        return self.model(input_data)

