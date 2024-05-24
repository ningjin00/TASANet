from torch.nn import Module
from modules.ConvBlock import ConvBlock
from modules.ConvBlockIN import ConvBlockIN
from modules.TiedAttention import TiedAttention
class FusionBlock(Module):
    def __init__(self, channel_size=32):
        super().__init__()
        self.conv_in = ConvBlockIN(channel_size)
        self.conv1_sar = ConvBlock(in_size=channel_size, out_size=channel_size)
        self.conv2_sar = ConvBlock(in_size=channel_size, out_size=channel_size)
        self.conv3_sar = ConvBlock(in_size=channel_size, out_size=channel_size)
        self.conv4_sar = ConvBlock(in_size=channel_size, out_size=channel_size)
        self.conv5_sar = ConvBlock(in_size=channel_size, out_size=channel_size)
        self.conv1_ors = ConvBlock(in_size=channel_size, out_size=channel_size)
        self.conv2_ors = ConvBlock(in_size=channel_size, out_size=channel_size)
        self.conv3_ors = ConvBlock(in_size=channel_size, out_size=channel_size)
        self.conv4_ors = ConvBlock(in_size=channel_size, out_size=channel_size)
        self.conv5_ors = ConvBlock(in_size=channel_size, out_size=channel_size)
        self.na1 = TiedAttention(channel_size)
        self.na2 = TiedAttention(channel_size)
        self.na3 = TiedAttention(channel_size)
        self.conv_1 = ConvBlock(in_size=channel_size , out_size=channel_size)
        self.conv_2 = ConvBlock(in_size=channel_size, out_size=channel_size)

    def forward(self, x):
        x = self.conv_in(x)
        sar = self.conv1_sar(x[0])
        ors = self.conv1_ors(x[1])
        sar = self.conv2_sar(sar)
        ors = self.conv2_ors(ors)
        n1 = self.na1([sar, ors])
        sar = self.conv3_sar(sar)
        ors = self.conv3_ors(ors)
        n2 = self.na2([sar, ors])
        sar = self.conv4_sar(sar)
        ors = self.conv4_ors(ors)
        n3 = self.na3([sar, ors])
        sar = self.conv5_sar(sar)
        ors = self.conv5_ors(ors)
        out = self.conv_1(sar+ors)
        out = self.conv_2(out + n1 + n2 + n3)

        return out
