import torch
import torch.nn as nn
import torchvision

from TCN import TemporalConvNet


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ResTCN(nn.Module):
    def __init__(self):
        super(ResTCN, self).__init__()

        self.spatial_feat_dim = 32
        self.num_classes = 4
        self.nhid = 128
        self.levels = 8
        self.kernel_size = 7
        self.dropout = .1
        self.channel_sizes = [self.nhid] * self.levels

        self.tcn = TemporalConvNet(
            self.spatial_feat_dim,
            self.channel_sizes,
            kernel_size=self.kernel_size,
            dropout=self.dropout)
        self.linear = nn.Linear(self.channel_sizes[-1], self.num_classes)

        self.model_conv = torchvision.models.resnet18(pretrained=True)
        # for param in self.model_conv.parameters():
        #     param.requires_grad = False

        num_ftrs = self.model_conv.fc.in_features
        # self.model_conv.fc = nn.Linear(num_ftrs, 4)
        self.model_conv.fc = nn.Linear(num_ftrs, self.spatial_feat_dim)
        # self.model_conv.fc = Identity()

        # self.rnn = nn.LSTM(self.spatial_feat_dim, 64, 1, batch_first=True)
        # self.linear = nn.Linear(64, 4)

    def forward(self, data):
        # t = 0
        # x = data[:, t, :, :, :]
        # output = self.model_conv(x)

        z = torch.zeros([data.shape[0], data.shape[1], self.spatial_feat_dim]).cuda()
        for t in range(data.size(1)):
            x = self.model_conv(data[:, t, :, :, :])
            z[:, t, :] = x

        # y, _ = self.rnn(z)
        # output = self.linear(torch.sum(y, dim=1))

        z = z.transpose(1, 2)
        y = self.tcn(z)
        # output = self.linear(y[:, :, -1])
        output = self.linear(torch.sum(y, dim=2))

        return output
