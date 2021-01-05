import os

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

TRAIN_PATH = "D:\\Datasets\\CyberProject\\"
MODEL_PATH = "./network.pth"
EPOCHS = 3
DEVICE = None


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # https://stackoverflow.com/questions/53784998/how-are-the-pytorch-dimensions-for-linear-layers-calculated
        self.fc1 = nn.Linear(16 * 22 * 22, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 22 * 22)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def main():
    global DEVICE
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = ImageFolder(TRAIN_PATH, transform=transform)
    trainloader = DataLoader(trainset, batch_size=4,
                             shuffle=True, num_workers=4)

    net = Net()
    # net.load_state_dict(torch.load(MODEL_PATH))
    net.to(DEVICE)
    train(net, trainloader)


def train(net, trainloader):
    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.05)

    print("Started training")

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = lossFunction(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 50 == 49:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    torch.save(net.state_dict(), MODEL_PATH)


if __name__ == "__main__":
    main()
