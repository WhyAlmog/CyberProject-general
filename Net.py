import torch.nn.functional as F
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # https://stackoverflow.com/questions/53784998/how-are-the-pytorch-dimensions-for-linear-layers-calculated
        self.fc1 = nn.Linear(16 * 22 * 22, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 96)
        self.fc4 = nn.Linear(96, 48)
        self.fc5 = nn.Linear(48, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 22 * 22)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
