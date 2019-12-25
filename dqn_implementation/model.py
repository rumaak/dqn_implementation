import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, action_count):
        super().__init__()
        self.conv_first = nn.Conv2d(4, 32, 8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv_second = nn.Conv2d(32, 64, 4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv_third = nn.Conv2d(64, 64, 3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.dense_first = nn.Linear(3136, 512)
        self.dense_second = nn.Linear(512, action_count)

    def forward(self, inp):
        # inp = batch x channels x width x height
        x = F.relu(self.bn1(self.conv_first(inp)))
        x = F.relu(self.bn2(self.conv_second(x)))
        x = F.relu(self.bn3(self.conv_third(x)))

        x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.dense_first(x))
        return self.dense_second(x)
