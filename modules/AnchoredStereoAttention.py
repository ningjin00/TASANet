from torch.nn import Module, Softmax,Conv2d
class AnchoredStereoAttention(Module):
    def __init__(self, channel_size):
        super().__init__()
        self.conv_query = Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=3, stride=1, padding=1)
        self.conv_key =  Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=3, stride=1, padding=1)
        self.conv_anchor =  Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=3, stride=1, padding=1)
        self.conv_value =  Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=3, stride=1, padding=1)
        self.md_softmax = Softmax(dim=1)
        self.me_softmax = Softmax(dim=1)
        self.d = 16.0

    def forward(self, x):
        anchor = self.conv_anchor(x)
        me = (self.conv_query(x) @ anchor.transpose(-1, -2)).transpose(-1, -2)
        md = (self.conv_key(x) @ anchor.transpose(-1, -2)).transpose(-1, -2)
        return (self.conv_value(x) @ self.md_softmax(md / self.d)) @ self.me_softmax(me / self.d)
