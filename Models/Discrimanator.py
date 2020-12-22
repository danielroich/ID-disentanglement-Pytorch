import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, num_features, n_hid, dropout = 0.25):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(num_features, n_hid),
            nn.LeakyReLU(),
            nn.BatchNorm1d(n_hid, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(dropout),
            nn.Linear(n_hid, n_hid // 2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(n_hid // 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(dropout),
            nn.Linear(n_hid // 2, n_hid // 4),
            nn.LeakyReLU(),
            nn.BatchNorm1d(n_hid // 4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(dropout),
            nn.Linear(n_hid // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        return self.main(input_data)

