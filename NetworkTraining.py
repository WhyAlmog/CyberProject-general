import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from Net import Net

TRAIN_PATH = "C:\\Users\\almog\\OneDrive\\VSCodeWorkspace\\CyberProject\\CyberProject-data\\datasetProcessed\\train\\"
MODEL_PATH = "./network.pth"
EPOCHS = 3
DEVICE = None
LEARNING_RATE = 0.0001
BATCH_SIZE = 4


def main():
    global DEVICE
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = ImageFolder(TRAIN_PATH, transform=transform)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=0)

    net = Net()
    # net.load_state_dict(torch.load(MODEL_PATH))
    net.train()
    net.to(DEVICE)
    train(net, trainloader)


def train(net, trainloader):
    """train the network

    Args:
        net: network
        trainloader: training data
    """
    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

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
