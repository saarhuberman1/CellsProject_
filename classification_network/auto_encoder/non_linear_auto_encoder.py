# from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch


class Encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels, ds=True):
        super(Encoder_block, self).__init__()
        self._conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self._relu1 = nn.ReLU()
        self._bn1 = nn.BatchNorm2d(out_channels)
        self._conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self._relu2 = nn.ReLU()
        self._bn2 = nn.BatchNorm2d(out_channels)
        if ds:
            self._conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1,
                                    stride=(2,2))
            self._relu3 = nn.ReLU()
            self._bn3 = nn.BatchNorm2d(out_channels)
        self.ds = ds

    def forward(self, x):
        x = self._relu1(self._bn1(self._conv1(x)))
        x = self._relu2(self._bn2(self._conv2(x)))
        if self.ds:
            x = self._relu3(self._bn3(self._conv3(x)))
        return x

class Decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder_block, self).__init__()
        self._conv1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=(2,2))
        self._relu1 = nn.ReLU()
        self._bn1 = nn.BatchNorm2d(out_channels)
        self._conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self._relu2 = nn.ReLU()
        self._bn2 = nn.BatchNorm2d(out_channels)
        self._conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self._relu1(self._bn1(self._conv1(x)))
        x = self._relu2(self._bn2(self._conv2(x)))
        x = self._conv3(x)
        return x


class AutoEncoder(nn.Module):
    """
    A Fully Convolutional EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNetFCNN.from_pretrained('efficientnet-b0')

    """
    def __init__(self, in_channels=3, expansion=32):
        super(AutoEncoder, self).__init__()
        self.exapnsion = expansion

        # encoder
        self._conv1 = Encoder_block(in_channels=in_channels, out_channels=expansion)
        self._conv2 = Encoder_block(in_channels=expansion, out_channels=expansion)
        self._conv3 = Encoder_block(in_channels=expansion, out_channels=expansion)
        self._conv4 = Encoder_block(in_channels=expansion, out_channels=expansion)
        self._conv5 = Encoder_block(in_channels=expansion, out_channels=expansion, ds=False)
        # self._conv6 = nn.Conv2d(in_channels=expansion, out_channels=expansion, kernel_size=3, stride=(2, 2))
        # self._conv7 = nn.Conv2d(in_channels=expansion, out_channels=expansion, kernel_size=3)
        self._encoder = nn.ModuleList([self._conv1, self._conv2, self._conv3, self._conv4, self._conv5])
            #, self._conv6, self._conv7])

        # decoder:
        self._conv8 = Decoder_block(in_channels=expansion, out_channels=expansion)
        self._conv9 = Decoder_block(in_channels=expansion, out_channels=expansion)
        self._conv10 = Decoder_block(in_channels=expansion, out_channels=expansion)
        self._conv11 = Decoder_block(in_channels=expansion, out_channels=in_channels)
        # self._conv12 = nn.ConvTranspose2d(in_channels=expansion, out_channels=expansion, kernel_size=2, stride=2)
        # self._conv13 = nn.Conv2d(in_channels=expansion, out_channels=in_channels, kernel_size=3)
        self._decoder = nn.ModuleList([self._conv8, self._conv9, self._conv10, self._conv11])
                # self._conv12, self._conv13])

    def forward(self, x):
        for e in self._encoder:
            x = e(x)

        for d in self._decoder:
            x = d(x)

        return x

    def encode(self, x):
        for e in self._encoder:
            x = e(x)

        return x

    def decode(self, x):
        for d in self._decoder:
            x = d(x)

        return x

    def get_num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = AutoEncoder()
    dummy_input = torch.randn(1, 3, 256, 256)
    # res = model.forward(dummy_input)
    # print(res)
    # dummy_input = torch.randn(1, 3, 1440, 1440)
    res = model.forward(dummy_input)
    print(res.shape)
    flag = 0