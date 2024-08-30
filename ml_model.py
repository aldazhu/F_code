import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class MyTemporalConvNet(nn.Module):
    def __init__(self, num_inputs,num_outputs, num_channels, kernel_size=2, dropout=0.2):
        super(MyTemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_outputs)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.network(x)
        
        x = x[:, :, -1] # get the last layer
        
        return self.fc(x)
    
class MLP(nn.Module):
    def __init__(self, input_features, pre_days, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_features*pre_days, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = self.flatten(x)
        # print(f"x shape: {x.shape}")
        # print(f"x: {x}")
        logits = self.linear_relu_stack(x)
        # print(f"logits shape: {logits.shape}")
        # print(f"logits: {logits}")
        # input("press any key to continue")
        return logits
    
def demo_of_MLP_model():
    feature_num = 5 # OHLCV
    pre_days = 30
    hidden_dim = 512
    output_dim = 1
    model = MLP(feature_num, pre_days, hidden_dim, output_dim)
    print(model)
    print(f"model parameters: {sum(p.numel() for p in model.parameters())}")

    x = torch.randn(2, feature_num, pre_days)
    y = model(x)

    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")

def demo_of_TCN_model():
    feature_num = 5 # OHLCV
    num_outputs = 10
    sequence_length = 30
    batch_size = 2
    num_channels = [32,32,32,32,32]
    kernel_size = 2
    dropout = 0.2
    model = TemporalConvNet(feature_num, num_channels, kernel_size, dropout)
    model_stock = MyTemporalConvNet(feature_num, num_outputs, num_channels, kernel_size, dropout)
    print(model)
    print(f"model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"model_stock parameters: {sum(p.numel() for p in model_stock.parameters())}")

    x = torch.randn(batch_size, feature_num, sequence_length)
    y = model(x)

    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")

    y = model_stock(x)
    print(f"y shape: {y.shape}")
    
if __name__ == "__main__":
    # demo_of_TCN_model()
    demo_of_MLP_model()

    
