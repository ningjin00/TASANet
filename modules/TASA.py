from torch.nn import Module
from modules.ConvBlockOUT import ConvBlockOUT
from modules.FusionBlock import FusionBlock
from modules.RestorationStage import RestorationStage
class TASA(Module):
    def __init__(self, channel_size=32,recover_block=3,three_layer=3):
        super().__init__()
        self.fusion = FusionBlock(channel_size=channel_size)
        self.recover_module = RestorationStage(channel_size=channel_size,recover_block=recover_block,three_layer=three_layer)
        self.conv_out = ConvBlockOUT(channel_size=channel_size)

    def forward(self, x):
        x = self.fusion(x)
        x = self.recover_module(x)
        return self.conv_out(x)