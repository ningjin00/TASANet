from torch.nn import Linear,ReLU,Dropout,Module,init
class Mlp(Module):
    def __init__(self,feature_size=256):
        super( ).__init__()
        self.fc1 = Linear(feature_size, feature_size*2)  #
        self.fc2 = Linear(feature_size*2, feature_size)  #
        self.act_fn = ReLU(True)
        self.dropout = Dropout(0.1)
        self._init_weights()

    def _init_weights(self):
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.normal_(self.fc1.bias, std=1e-6)
        init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)  #
        x = self.act_fn(x)
        x = self.dropout(x)  #
        x = self.fc2(x)  #
        x = self.dropout(x)
        return x
