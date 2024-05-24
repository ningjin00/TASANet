from torch.nn import Module, Conv2d, LayerNorm, Dropout, ReLU
class ConvBlock(Module):
    def __init__(self, in_size, out_size, feature_size=256):
        super().__init__()
        self.conv_1 = Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, stride=1, padding=1)
        self.layer = LayerNorm([feature_size, feature_size], eps=1e-6)
        self.drop_1 = Dropout(p=0.1)
        self.act_1 = ReLU(True)
        self.conv_2 = Conv2d(in_channels=out_size, out_channels=out_size, kernel_size=3, stride=1, padding=1)
        self.act_2 = ReLU(True)
    def forward(self, x):
        x = self.act_1(self.drop_1(self.layer(self.conv_1(x))))
        return self.act_2(self.conv_2(x))
