import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self, num_features, dropout=0.25, n_hid=128):
        super().__init__()
        self.model = nn.Sequential(
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
            nn.Linear(n_hid // 4, 512)
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_tensor):
        return self.model(input_tensor)
