import torch.nn.functional as F
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 24, 5)
        self.conv2 = nn.Conv2d(24, 24, 5)
        self.conv3 = nn.Conv2d(24, 24, 3)
        self.fc1 = nn.Linear(24 * 10 ** 2, 4096)
        self.fc2 = nn.Linear(4096, 512)
        self.fc3 = nn.Linear(512, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 24 * 10 ** 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
