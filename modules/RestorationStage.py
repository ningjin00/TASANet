from torch.nn import Module, Sequential
from torch import cat
from modules.ConvBlock import ConvBlock
from modules.RestorationBlock import RestorationBlock
class RestorationStage(Module):
    def __init__(self, channel_size=32, recover_block=3, three_layer=3):
        super().__init__()
        m = []
        for i in range(recover_block):
            m.append(RestorationBlock(channel_size=channel_size, three_layer=three_layer))
        self.recover_stage = Sequential(*m)
        self.conv_1 = ConvBlock(channel_size * 2, channel_size)

    def forward(self, x):
        out = self.recover_stage(x)
        out = self.conv_1(cat([out, x], dim=1))
        return out
