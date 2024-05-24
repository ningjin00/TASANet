from torch.nn import Module, Softmax, Conv2d, AdaptiveAvgPool2d, AdaptiveMaxPool2d,ReLU
class ChannelAttention(Module):
    def __init__(self, channel_size=32, feature_size=256):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2d((feature_size, feature_size))
        self.max_pool = AdaptiveMaxPool2d((feature_size, feature_size))
        self.fc1 = Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=3, stride=1, padding=1)
        self.relu1 = ReLU(True)
        self.fc2 = Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=3, stride=1, padding=1)
        self.fc3 = Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=3, stride=1, padding=1)
        self.relu3 = ReLU(True)
        self.fc4 = Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=3, stride=1, padding=1)
        self.attn_sigmoid = Softmax()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        max_pool_out = self.max_pool(x)
        avg_out = self.fc2(self.relu1(self.fc1(avg_pool_out)))
        max_out = self.fc2(self.relu1(self.fc1(max_pool_out)))
        out = avg_out + max_out
        sigmoid_out = self.attn_sigmoid(out)
        channel_attn_out = x @ sigmoid_out.transpose(-1, -2)
        return channel_attn_out
