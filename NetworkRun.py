import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from Net import Net

TRAIN_PATH = "C:\\Users\\Manor\\OneDrive\\VSCodeWorkspace\\CyberProject\\CyberProject-data\\datasetProcessed\\train\\"
TEST_PATH = "C:\\Users\\Manor\\OneDrive\\VSCodeWorkspace\\CyberProject\\CyberProject-data\\datasetProcessed\\test\\"
MODEL_PATH = "./network.pth"
EPOCHS = 3
DEVICE = None
LEARNING_RATE = 0.0001
BATCH_SIZE = 4


def main():
    """train and then test the network, more information can be found at the individual train/test files
    """
    global DEVICE
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = ImageFolder(TRAIN_PATH, transform=transform)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=0)

    testset = ImageFolder(TEST_PATH, transform=transform)
    testloader = DataLoader(testset, batch_size=4,
                            shuffle=True, num_workers=0)

    net = Net()
    net.train()
    net.to(DEVICE)
    train(net, trainloader)

    net.eval()
    test(net, testloader)

    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    torch.save(net.state_dict(), MODEL_PATH)


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
        for data in trainloader:
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = lossFunction(outputs, labels)
            loss.backward()
            optimizer.step()


def test(net, trainloader):
    """test the performance of the network

    Args:
        net: network
        trainloader: testing data
    """
    print("Started testing")
    choices = 0
    correct_choices = 0

    for data in trainloader:
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

        outputs = net(inputs)

        for i, label in enumerate(labels):
            choices += 1

            if torch.argmax(outputs[i]) == label:
                correct_choices += 1

    print("Accuracy: {:.3f}".format(correct_choices / choices))


if __name__ == "__main__":
    main()
