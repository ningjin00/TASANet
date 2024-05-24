from torch.nn import Module,Sequential
from torch import cat
from modules.ConvBlock import ConvBlock
from modules.HierarchicalModeling import HierarchicalModeling
class RestorationBlock(Module):
    def __init__(self, channel_size=32,three_layer=3):
        super().__init__()
        m= []
        for i in range(three_layer):
            m.append(HierarchicalModeling(channel_size=channel_size))
        self.recover_block = Sequential(*m)
        self.conv_1 = ConvBlock(channel_size*2 , channel_size)

    def forward(self, x):
        out = self.recover_block(x)
        out = self.conv_1(cat([out, x], dim=1))
        return out