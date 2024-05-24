from torch.nn import Module, ReLU,Conv2d
class ConvBlockOUT(Module):
    def __init__(self, channel_size=32):
        super().__init__()
        self.conv_1 = Conv2d(in_channels=channel_size, out_channels=13, kernel_size=3, stride=1, padding=1)
        self.act_1 = ReLU(True)
        self.conv_2=Conv2d(in_channels=13, out_channels=13, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        x = self.conv_1(x)
        return self.conv_2(self.act_1(x))