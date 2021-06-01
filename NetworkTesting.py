import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import torch.nn.functional as F

from Net import Net

TEST_PATH = "C:\\Users\\Manor\\OneDrive\\VSCodeWorkspace\\CyberProject\\CyberProject-data\\datasetProcessed\\test\\"
MODEL_PATH = "./network.pth"
DEVICE = None


def main():
    global DEVICE
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    transform = transforms.Compose([transforms.ToTensor()])
    testset = ImageFolder(TEST_PATH, transform=transform)
    testloader = DataLoader(testset, batch_size=4,
                            shuffle=True, num_workers=0)

    net = Net()
    net.load_state_dict(torch.load(MODEL_PATH))
    net.eval()
    net.to(DEVICE)
    test(net, testloader)


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
