import torch.nn.functional as F
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 256, 7)
        self.conv2 = nn.Conv2d(256, 128, 5)
        self.conv3 = nn.Conv2d(128, 64, 3)
        self.fc1 = nn.Linear(64 * 9 ** 2, 1024)
        self.fc2 = nn.Linear(1024, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 9 ** 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
