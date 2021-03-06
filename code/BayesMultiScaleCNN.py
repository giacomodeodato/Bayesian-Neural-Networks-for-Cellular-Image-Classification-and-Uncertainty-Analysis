import torch
import torch.nn as nn
import torch.nn.functional as F
import vinn

class Pathway(vinn.Module):

    def __init__(self, in_channels, out_channels, post_pool, pre_pool=None):
        super(Pathway, self).__init__()

        self.pre_pool = pre_pool
        self.post_pool = post_pool

        self.conv1 = vinn.Conv2d(in_channels, out_channels, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = vinn.Conv2d(out_channels, out_channels, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = vinn.Conv2d(out_channels, out_channels, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.pre_pool is not None:
            x = F.max_pool2d(x, self.pre_pool, self.pre_pool)
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.max_pool2d(x, self.post_pool, self.post_pool)
        return x

class BayesMultiScaleCNN(vinn.Module):

    def __init__(self, n_outputs, in_channels=3, n_features=1024):
        super(BayesMultiScaleCNN, self).__init__()

        self.pathway1 = Pathway(in_channels, 16, 64 )
        self.pathway2 = Pathway(in_channels, 16, 32, 2)
        self.pathway3 = Pathway(in_channels, 16, 16, 4)
        self.pathway4 = Pathway(in_channels, 32, 8,  8)
        self.pathway5 = Pathway(in_channels, 32, 4,  16)
        self.pathway6 = Pathway(in_channels, 32, 2,  32)

        self.conv = vinn.Conv2d(144, n_features, 1)
        self.bn = nn.BatchNorm1d(n_features)
        self.dense = vinn.Linear(n_features, n_outputs)
        
    def forward(self, x):
        output = torch.cat(
            (
                self.pathway1(x), 
                self.pathway2(x), 
                self.pathway3(x), 
                self.pathway4(x), 
                self.pathway5(x), 
                self.pathway6(x)
            ),
            dim=1
        )
        output = F.avg_pool2d(F.relu(self.conv(output)), (16,20), 1)
        output = torch.flatten(output, 1, -1)
        output = self.bn(output)
        output = self.dense(output)
        return output