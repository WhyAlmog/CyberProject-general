import torch.nn.functional as F
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 56, 5)
        self.conv2 = nn.Conv2d(56, 42, 5)
        self.conv3 = nn.Conv2d(42, 32, 3)
        
        self.fc1 = nn.Linear(32 * 10 ** 2, 4096)
        self.fc2 = nn.Linear(4096, 3)

    def forward(self, x):
        #! for every layer (division is always rounded down):
        #! outputSize = outputNeurons * (inputSize - (layerKernelSize / 2))
        #! applying a pool halves the size
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # we can use 10 ** 2 because the images are squares
        x = x.view(-1, 32 * 10 ** 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
