from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch

class EfficientNetFCNN(nn.Module):
    """
    A Fully Convolutional EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNetFCNN.from_pretrained('efficientnet-b0')

    """
    def __init__(self, name='efficientnet-b0', image_size=None):
        super(EfficientNetFCNN, self).__init__()
        # self._efficient_backbone = EfficientNet.from_pretrained(name, image_size=image_size)
        self._efficient_backbone = EfficientNet.from_name(name)
        self.name=name

        # create fully convolutional head
        image_size = self._efficient_backbone.get_image_size(name)
        dummy_input = torch.randn(1, 3, image_size, image_size)
        self._efficient_backbone.eval()
        dummy_features = self._efficient_backbone.extract_features(dummy_input)
        self._efficient_backbone.train()
        b, f_c, f_w, f_h = dummy_features.shape
        self._gap = nn.AvgPool2d((f_w, f_h), stride=1)
        self._final_1x1_conv = nn.Conv2d(in_channels=f_c, out_channels=2, kernel_size=1)

        # clean old FC head:
        self._efficient_backbone._avg_pooling = None
        self._efficient_backbone._dropout = None
        self._efficient_backbone._fc = None

    def forward(self, x):
        x = self._efficient_backbone.extract_features(x)
        x = self._gap(x)
        x = self._final_1x1_conv(x)
        return x

    def get_im_size(self):
        return self._efficient_backbone.get_image_size(self.name)

    def get_num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = EfficientNetFCNN(name='efficientnet-b0')
    # dummy_input = torch.randn(1, 3, 224, 224)
    # res = model.forward(dummy_input)
    # print(res)
    dummy_input = torch.randn(1, 3, 1440, 1440)
    res = model.forward(dummy_input)
    print(res)
    flag = 0