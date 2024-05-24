from torch.nn import Module, Conv2d, Dropout,Softmax

class WindowAttention(Module):
    def __init__(self, channel_size=32):
        super().__init__()
        self.query_dense = Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=3, stride=1, padding=1)
        self.key_dense = Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=3, stride=1, padding=1)
        self.value_dense = Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=3, stride=1, padding=1)
        self.b_conv = Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=7, stride=1, padding=3)
        self.d = 16.0
        self.soft = Softmax()
        self.drop_attn = Dropout(0.1)

    def forward(self, x):
        attn = self.drop_attn(self.query_dense(x) @ self.key_dense(x).transpose(-1, -2) / self.d + self.b_conv(x))
        return self.value_dense(x) @ self.soft(attn).transpose(-1, -2)
