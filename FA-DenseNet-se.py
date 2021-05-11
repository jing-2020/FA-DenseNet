# codingutf-8

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

torch.manual_seed(88)
torch.cuda.manual_seed(88)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class block(nn.Module):
    def __init__(self, in_features, hidden1, hidden2, hidden3, out_features):
        super(block, self).__init__()
        self.short = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.ReLU()
        )
        self.density = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, out_features)
            , nn.Dropout(0.5)
        )

    def forward(self, x):
        x = self.density(self.short(x))
        return x


class Fullnet(nn.Module):
    def __init__(self, in_features, out_features):
        super(Fullnet, self).__init__()
        self.input = nn.Linear(in_features, 68)
        self.block1 = block(68, 113, 281, 83, 68)
        self.block2 = block(68, 138, 283, 128, 68)
        self.block3 = block(68, 160, 315, 139, 68)
        self.block4 = block(68, 120, 231, 107, 68)
        self.block5 = block(68, 99, 151, 139, out_features)
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.input(x)
        d1 = x
        x = self.block1(d1)
        d2 = x + d1
        x = self.block2(d2)
        d3 = x + d1 + d2
        x = self.block3(d3)
        d4 = x + d1 + d2 + d3
        x = self.block4(d4)
        d5 = x + d1 + d2 + d3 + d4
        x = self.block5(d5)
        x = self.output(x)
        return x


if __name__ == "__main__":

    model = Fullnet(24, 24).to(device)
    print(model)
