from NetworkTraining import MODEL_PATH
import socket
import threading
import time

from EV3Commands import *

from Utils import *

from Net import Net

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

CLASSES = {
    0: "paper",
    1: "plastic",
    2: "tin"
}

TUBE_MOVEMENT = {
    0: tube_left_position,
    1: tube_middle_position,
    2: tube_right_position
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
        res = wait_touch_sensor_clicked(EV3)
        if res == "success":
            send(PHONE, "TAKE PICTURE")

            filename = "/Time" + str(time.time()) + ".jpg"
            with open(FOLDER + filename, 'wb') as f:
                f.write(receive(PHONE))

            tube_pos = network_eval(filename)

            TUBE_MOVEMENT[tube_pos](EV3)
            open_trapdoor(EV3)
            time.sleep(1.5)
            close_trapdoor(EV3)
            tube_middle_position(EV3)


def network_eval(filename) -> int:
    image = Image.open(FOLDER + filename)

    x = TF.to_tensor(image)
    # create a fake batch
    x.unsqueeze_(0)

    output = NETWORK(x)

    _, predicted = torch.max(output, 1)

    print(CLASSES[predicted.numpy()[0]])
    return predicted.numpy()[0]


def exit(phone_server: socket):
    status = receive_string(EV3_EXIT)
    print(status)

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
