from torchvision.models import *
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms


def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding='same'):
    """3x3 convolution with padding"""
    if padding == 'same':
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False, dilation=dilation)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,
                 kernel_size=3,
                 norm_layer=None):
        super(ResBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.shortcut_conv = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
        self.conv1 = nn.Conv2d(inplanes, planes // 2, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(planes // 2, planes // 2, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.conv3 = nn.Conv2d(planes // 2, planes, kernel_size=1, stride=1, padding=0)

        self.normalizer_fn = norm_layer(planes)
        self.activation_fn = nn.ReLU(inplace=True)

        self.stride = stride
        self.out_planes = planes

    def forward(self, x):
        shortcut = x
        (_, _, _, x_planes) = x.size()

        if self.stride != 1 or x_planes != self.out_planes:
            shortcut = self.shortcut_conv(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x += shortcut
        x = self.normalizer_fn(x)
        x = self.activation_fn(x)

        return x


class ResFCN256(nn.Module):
    def __init__(self, resolution_input=256, resolution_output=256, channel=3, size=16):
        super().__init__()
        self.input_resolution = resolution_input
        self.output_resolution = resolution_output
        self.channel = channel
        self.size = size

        # Encoder
        self.block0 = conv3x3(in_planes=3, out_planes=self.size, padding='same')
        self.block1 = ResBlock(inplanes=self.size, planes=self.size * 2, stride=2)
        self.block2 = ResBlock(inplanes=self.size * 2, planes=self.size * 2, stride=1)
        self.block3 = ResBlock(inplanes=self.size * 2, planes=self.size * 4, stride=2)
        self.block4 = ResBlock(inplanes=self.size * 4, planes=self.size * 4, stride=1)
        self.block5 = ResBlock(inplanes=self.size * 4, planes=self.size * 8, stride=2)
        self.block6 = ResBlock(inplanes=self.size * 8, planes=self.size * 8, stride=1)
        self.block7 = ResBlock(inplanes=self.size * 8, planes=self.size * 16, stride=2)
        self.block8 = ResBlock(inplanes=self.size * 16, planes=self.size * 16, stride=1)
        self.block9 = ResBlock(inplanes=self.size * 16, planes=self.size * 32, stride=2)
        self.block10 = ResBlock(inplanes=self.size * 32, planes=self.size * 32, stride=1)

        # Decoder
        self.upsample0 = nn.ConvTranspose2d(self.size * 32, self.size * 32, kernel_size=3, stride=1,
                                            padding=1)  # keep shape invariant.
        self.upsample1 = nn.ConvTranspose2d(self.size * 32, self.size * 16, kernel_size=4, stride=2,
                                            padding=1)  # half downsample.
        self.upsample2 = nn.ConvTranspose2d(self.size * 16, self.size * 16, kernel_size=3, stride=1,
                                            padding=1)  # keep shape invariant.
        self.upsample3 = nn.ConvTranspose2d(self.size * 16, self.size * 16, kernel_size=3, stride=1,
                                            padding=1)  # keep shape invariant.

        self.upsample4 = nn.ConvTranspose2d(self.size * 16, self.size * 8, kernel_size=4, stride=2,
                                            padding=1)  # half downsample.
        self.upsample5 = nn.ConvTranspose2d(self.size * 8, self.size * 8, kernel_size=3, stride=1,
                                            padding=1)  # keep shape invariant.
        self.upsample6 = nn.ConvTranspose2d(self.size * 8, self.size * 8, kernel_size=3, stride=1,
                                            padding=1)  # keep shape invariant.

        self.upsample7 = nn.ConvTranspose2d(self.size * 8, self.size * 4, kernel_size=4, stride=2,
                                            padding=1)  # half downsample.
        self.upsample8 = nn.ConvTranspose2d(self.size * 4, self.size * 4, kernel_size=3, stride=1,
                                            padding=1)  # keep shape invariant.
        self.upsample9 = nn.ConvTranspose2d(self.size * 4, self.size * 4, kernel_size=3, stride=1,
                                            padding=1)  # keep shape invariant.

        self.upsample10 = nn.ConvTranspose2d(self.size * 4, self.size * 2, kernel_size=4, stride=2,
                                             padding=1)  # half downsample.
        self.upsample11 = nn.ConvTranspose2d(self.size * 2, self.size * 2, kernel_size=3, stride=1,
                                             padding=1)  # keep shape invariant.

        self.upsample12 = nn.ConvTranspose2d(self.size * 2, self.size, kernel_size=4, stride=2,
                                             padding=1)  # half downsample.
        self.upsample13 = nn.ConvTranspose2d(self.size, self.size, kernel_size=3, stride=1,
                                             padding=1)  # keep shape invariant.

        self.upsample14 = nn.ConvTranspose2d(self.size, self.channel, kernel_size=3, stride=1,
                                             padding=1)  # keep shape invariant.
        self.upsample15 = nn.ConvTranspose2d(self.channel, self.channel, kernel_size=3, stride=1,
                                             padding=1)  # keep shape invariant.
        self.upsample16 = nn.ConvTranspose2d(self.channel, self.channel, kernel_size=3, stride=1,
                                             padding=1)  # keep shape invariant.

        # ACT
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se = self.block0(x)  # 256 x 256 x 16
        se = self.block1(se)  # 128 x 128 x 32
        se = self.block2(se)  # 128 x 128 x 32
        se = self.block3(se)  # 64 x 64 x 64
        se = self.block4(se)  # 64 x 64 x 64
        se = self.block5(se)  # 32 x 32 x 128
        se = self.block6(se)  # 32 x 32 x 128
        se = self.block7(se)  # 16 x 16 x 256
        se = self.block8(se)  # 16 x 16 x 256
        se = self.block9(se)  # 8 x 8 x 512
        se = self.block10(se)  # 8 x 8 x 512

        pd = self.upsample0(se)  # 8 x 8 x 512
        pd = self.upsample1(pd)  # 16 x 16 x 256
        pd = self.upsample2(pd)  # 16 x 16 x 256
        pd = self.upsample3(pd)  # 16 x 16 x 256
        pd = self.upsample4(pd)  # 32 x 32 x 128
        pd = self.upsample5(pd)  # 32 x 32 x 128
        pd = self.upsample6(pd)  # 32 x 32 x 128
        pd = self.upsample7(pd)  # 64 x 64 x 64
        pd = self.upsample8(pd)  # 64 x 64 x 64
        pd = self.upsample9(pd)  # 64 x 64 x 64

        pd = self.upsample10(pd)  # 128 x 128 x 32
        pd = self.upsample11(pd)  # 128 x 128 x 32
        pd = self.upsample12(pd)  # 256 x 256 x 16
        pd = self.upsample13(pd)  # 256 x 256 x 16
        pd = self.upsample14(pd)  # 256 x 256 x 3
        pd = self.upsample15(pd)  # 256 x 256 x 3
        pos = self.upsample16(pd)  # 256 x 256 x 3

        pos = self.sigmoid(pos)
        return pos


class PRN:
    '''
        <Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network>
        This class serves as the wrapper of PRNet.
    '''

    def __init__(self, model_dir, **kwargs):
        # resolution of input and output image size.
        self.resolution_inp = kwargs.get("resolution_inp") or 256
        self.resolution_op = kwargs.get("resolution_op") or 256
        self.channel = kwargs.get("channel") or 3
        self.size = kwargs.get("size") or 16

        # 1) load model.
        self.pos_predictor = ResFCN256()
        state = torch.load(model_dir)
        self.pos_predictor.load_state_dict(state['prnet'])
        self.pos_predictor.eval()  # inference stage only.
        if torch.cuda.device_count() > 0:
            self.pos_predictor = self.pos_predictor.to("cuda")

    def net_forward(self, image):
        ''' The core of out method: regress the position map of a given image.
        Args:
            image: (3, 256, 256) array. value range: 0~1
        Returns:
            pos: the 3D position map. (3, 256, 256) array.
        '''
        return self.pos_predictor(image)

    def net_forward_loader(self, data_loader):
        with torch.no_grad():
            image, _ = next(iter(data_loader))
            if torch.cuda.device_count() > 0:
                image = image.to("cuda")
            output = self.net_forward(image)
        return output


def get_data(DATA_DIR, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], image_size = 256):
    full_dataset = dset.ImageFolder(root=DATA_DIR,
                           transform=transforms.Compose([
                                      transforms.Resize(image_size),
                                      # transforms.CenterCrop(image_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=mean, std=std),
                                  ]))
    return full_dataset

def make_loaders(dataset, batch_size):
    test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size)

    return test_loader


if __name__ == "__main__":
    prn = PRN(model_dir = 'Weights/latest.pth')

    DATA_DIR = r'Datasets/real_small/'  # looking in sub folder
    batch_size = 16

    data_folder = get_data(DATA_DIR)
    data_loader = make_loaders(data_folder, batch_size)
    output = prn.net_forward_loader(data_loader)

    print(output)
    print(output.shape)
