import torch
from .mobilefacenet import MobileFaceNet


class Encoder_Landmarks(torch.nn.Module):
    def __init__(self, model_dir='Weights/mobilefacenet_model_best.pth.tar'):
        super(Encoder_Landmarks, self).__init__()
        self.model = MobileFaceNet([112, 112], 136)
        checkpoint = torch.load(model_dir)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.eval()

    def forward(self, imgs):
        """
        without preprocess (face crop)
        :param imgs: shape: torch.Size([batch size, 3, 112, 112])
        :return:
        outputs - model results scaled to img size, shape: torch.Size([batch size, 136])
        landmarks - reshaped outputs + no jawline, shape: torch.Size([batch size, 51, 2])
        """
        outputs, _ = self.model(imgs)

        batch_size = imgs.shape[0]
        landmarks = torch.reshape(outputs*112, (batch_size, 68, 2))

        return outputs*112 , landmarks[:, 17:, :]
