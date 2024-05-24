from torch.nn import Module
from modules.AnchoredStereoAttention import AnchoredStereoAttention
from modules.ChannelAttention import ChannelAttention
from modules.ConvBlock import ConvBlock
from modules.MLP import Mlp
from modules.WindowAttention import WindowAttention
class HierarchicalModeling(Module):
    def __init__(self, channel_size=32, feature_size=256):
        super().__init__()
        self.anchored_attention = AnchoredStereoAttention(channel_size=channel_size)
        self.channel_attention = ChannelAttention(channel_size=channel_size, feature_size=feature_size)
        self.window_attention = WindowAttention(channel_size=channel_size)
        self.mpl = Mlp(feature_size=feature_size)
        self.conv_1 = ConvBlock(channel_size, channel_size, feature_size)

    def forward(self, x):
        out = self.mpl(x + self.anchored_attention(x) + self.channel_attention(x) + self.window_attention(x))
        out = self.conv_1(out)
        return out
