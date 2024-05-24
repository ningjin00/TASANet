from torch.nn import Module,AdaptiveMaxPool2d,AdaptiveAvgPool2d,Conv2d,ReLU,Softmax
from torch import cat
from modules.ConvBlock import ConvBlock
class TiedAttention(Module):
    def __init__(self, channel_size=32, feature_size=256):
        super().__init__()
        self.conv_1 = ConvBlock(in_size=channel_size*2, out_size=channel_size)
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
        x = self.conv_1(cat(x,dim=1))
        avg_pool_out = self.avg_pool(x)
        max_pool_out = self.max_pool(x)
        clash = self.fc2(self.relu1(self.fc1(avg_pool_out)))
        tie = self.fc4(self.relu3(self.fc3(max_pool_out)))
        mc = clash + tie
        sigmoid_out = self.attn_sigmoid(mc)
        out = x @ sigmoid_out.transpose(-1, -2)
        return out
