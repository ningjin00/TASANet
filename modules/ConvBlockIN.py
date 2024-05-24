from torch.nn import Module
from modules.ConvBlock import ConvBlock
class ConvBlockIN(Module):
    def __init__(self, channel_size=32):
        super().__init__()
        self.conv1_sar = ConvBlock(2, 8, 256)
        self.conv2_sar = ConvBlock(8, channel_size, 256)
        self.conv1_ors = ConvBlock(13, channel_size, 256)

    def forward(self, x):
        sar = self.conv2_sar(self.conv1_sar(x[:, 0:2, :, :]))
        ors = self.conv1_ors(x[:, 2:, :, :])
        return [sar, ors]
