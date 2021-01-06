from Network import MODEL_PATH
import socket
import threading
import time

from EV3Commands import WAIT_TOUCH_SENSOR_CLICKED
from EV3Commands import TEST_MOVE

from Net import Net

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch import nn

CLASSES = {
    0: "paper",
    1: "plastic",
    2: "tin"
}

NETWORK = None

FOLDER = "C:\\Users\\almog\\Desktop\\test"
MODEL_PATH = "./network.pth"

EV3_IP = "192.168.137.3"
EV3_PORT = 8070
EV3_EXIT_PORT = 8071
PHONE_SERVER_PORT = 9768

EV3 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
EV3_EXIT = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
PHONE = None

RUNNING = True


def send(conn: socket, data: str):
    """Sends data as bytes with a 4 byte header acting as the message's length (big endian)"""
    data = data.encode()
    conn.send(len(data).to_bytes(4, "big"))
    conn.send(data)


def receive(conn: socket):
    """Receives data according to the 4 bytes header protocol, see send function"""
    length = int.from_bytes(conn.recv(4), "big")
    return conn.recv(length)


def receive_string(conn: socket):
    """Receives data according to the 4 bytes header protocol, see send function"""
    length = int.from_bytes(conn.recv(4), "big")
    return conn.recv(length).decode()


def run():
    """
    workflow:
    wait for EV3 touch sensor to be pressed
    send request to capture image
    process returned image
    send appropriate motor commands to EV3
    repeat
    """
    while RUNNING:
        send(EV3, WAIT_TOUCH_SENSOR_CLICKED)
        if receive_string(EV3) == "success":
            send(PHONE, "TAKE PICTURE")

            filename = "/Time" + str(time.time()) + ".jpg"
            with open(FOLDER + filename, 'wb') as f:
                f.write(receive(PHONE))

            t = threading.Thread(target=network_eval, args=[filename])
            t.start()

            send(EV3, TEST_MOVE)
            receive_string(EV3)


def network_eval(filename):
    image = Image.open(FOLDER + filename)

    x = TF.to_tensor(image)
    # create a fake batch
    x.unsqueeze_(0)

    output = NETWORK(x)

    _, predicted = torch.max(output, 1)

    print(CLASSES[predicted.numpy()[0]])


def exit(phone_server: socket):
    status = receive_string(EV3_EXIT)
    print(status)
    if status == "exit":
        send(EV3, "exit")
    send(PHONE, "EXIT")

    EV3.close()
    PHONE.close()
    phone_server.close()


def main():
    global PHONE
    global NETWORK

    NETWORK = Net()
    NETWORK.load_state_dict(torch.load(MODEL_PATH))
    NETWORK.eval()
    print("Network loaded")

    EV3.connect((EV3_IP, EV3_PORT))
    EV3_EXIT.connect((EV3_IP, EV3_EXIT_PORT))
    print("EV3 Connected")

    phone_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    phone_server.bind(('0.0.0.0', PHONE_SERVER_PORT))
    phone_server.listen(5)

    # TODO add support for phone disconnecting due to network lag
    PHONE = phone_server.accept()[0]
    print("Phone Connected")

    t = threading.Thread(target=run)
    t.start()

    t = threading.Thread(target=exit, args=[phone_server])
    t.start()


if __name__ == "__main__":
    main()
