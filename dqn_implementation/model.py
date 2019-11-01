import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, action_count):
        super().__init__()
        self.conv_first = nn.Conv2d(4, 32, 8, stride=4)
        self.conv_second = nn.Conv2d(32, 64, 4, stride=2)
        self.conv_third = nn.Conv2d(64, 64, 3, stride=1)
        self.dense_first = nn.Linear(3136, 512)
        self.dense_second = nn.Linear(512, action_count)

    def forward(self, inp):
        # inp = batch x channels x width x height
        x = self.conv_first(inp)
        x = self.conv_second(x)
        x = self.conv_third(x)

        x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
        x = self.dense_first(x)
        x = self.dense_second(x)
        return x
