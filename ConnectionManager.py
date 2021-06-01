import socket
import threading
import time

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from EV3Commands import *
from Net import Net
from NetworkTraining import MODEL_PATH
from Utils import *

import os

# translate classes from network output to text
CLASSES = {
    0: "paper",
    1: "plastic",
    2: "tin"
}

# tranlate classes from network output to the correct tube function
TUBE_MOTOR_ROTATIONS = {
    0: tube_left_position,
    1: tube_middle_position,
    2: tube_right_position
}

# network file location
MODEL_PATH = "./network.pth"
NETWORK = None

# location to save the pictures to
IMAGE_FOLDER = "C:\\Users\\Manor\\OneDrive\\VSCodeWorkspace\\CyberProject\\CyberProject-data\\images\\"

IMAGE_DELETION_ERROR = "Failed to delete image: {}, overriting data instead"

EV3_IP = "192.168.137.3"
EV3_PORT = 8070
EV3_EXIT_PORT = 8071
PHONE_SERVER_PORT = 9768

EV3 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
EV3_EXIT = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
PHONE = None
PHONE_AES = None

RUNNING = True


def run():
    """
    The main body on the program. You can see more about the structure in the attached documentation file.
    """
    while RUNNING:
        # wait for the button to be clicked on the robot
        res = wait_touch_sensor_clicked(EV3)
        if res == "success":
            successful = False
            filepath = IMAGE_FOLDER + "temp.jpg"
            image = None

            while not successful:
                # take the picture and save it to disk
                send(PHONE, "TAKE PICTURE", PHONE_AES)
                with open(filepath, 'wb') as f:
                    f.write(receive(PHONE, PHONE_AES))

                # validate the image's integrity
                try:
                    image = Image.open(filepath)
                    image.load()
                    successful = True
                except Exception:
                    image.close()

                    # delete the image if it is broken
                    try:
                        os.remove(filepath)
                    except Exception:
                        print(IMAGE_DELETION_ERROR.format(filepath))

            network_prediction = network_eval(image)
            print(CLASSES[network_prediction])

            # rename the image to the network's prediction, will make it easier to expand the dataset later
            os.rename(filepath, IMAGE_FOLDER +
                      CLASSES[network_prediction] + str(int(time.time())) + ".jpg")

            # operate the tube on the robot
            TUBE_MOTOR_ROTATIONS[network_prediction](EV3)
            open_trapdoor(EV3)
            time.sleep(1)
            close_trapdoor(EV3)
            tube_middle_position(EV3)


def network_eval(image: Image) -> int:
    """use the network to decide the category of the trash in the picture

    Args:
        image (Image): image in PIL Image format, size of 100x100 pixles

    Returns:
        int: type of garbage, use the CLASSES dictionary for conversion
    """
    x = TF.to_tensor(image)
    # create a fake batch
    x.unsqueeze_(0)

    output = NETWORK(x)

    # get the most lit output neuron
    _, predicted = torch.max(output, 1)

    return predicted.numpy()[0]


def exit(phone_server: socket):
    """exit the program, send appropriate messages to all of the connected devices

    Args:
        phone_server (socket): the server listening for connection from the phone
    """
    global RUNNING

    status = receive_string(EV3_EXIT)
    print(status)

    send(PHONE, "EXIT", PHONE_AES)

    EV3.close()
    PHONE.close()
    phone_server.close()
    RUNNING = False


def main():
    """start the program, load the network and connect to the EV3 and the phone.
    """
    global PHONE
    global NETWORK
    global PHONE_AES

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

    PHONE = phone_server.accept()[0]
    print("Phone Connected")

    PHONE_AES = establish_connection(PHONE)
    print("encrypted")

    t = threading.Thread(target=run)
    t.start()

    t = threading.Thread(target=exit, args=[phone_server])
    t.start()


if __name__ == "__main__":
    main()
