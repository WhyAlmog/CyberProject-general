import os
import socket
import time

from PIL import Image

from Utils import *

from EV3Commands import wait_touch_sensor_clicked

EV3 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
EV3_IP = "192.168.137.3"
EV3_PORT = 8070

EV3_EXIT = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
EV3_EXIT_PORT = 8071

PHONE = None
PHONE_SERVER_PORT = 9768


IMAGE_FOLDER = "C:\\Users\\almog\\OneDrive\\VSCodeWorkspace\\CyberProject\\CyberProject-data\\images\\"


def run():
    while True:
        #wait_touch_sensor_clicked(EV3)
        input()

        successful = False
        filepath = IMAGE_FOLDER + str(int(time.time())) + ".jpg"
        image = None
        while not successful:
            try:
                send(PHONE, "TAKE PICTURE")
                with open(filepath, 'wb') as f:
                    f.write(receive(PHONE))

                image = Image.open(filepath)

                try:
                    image.load()
                    successful = True
                except Exception:
                    image.close()
            except Exception:
                successful = False

def main():
    global PHONE

    phone_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    phone_server.bind(('0.0.0.0', PHONE_SERVER_PORT))
    phone_server.listen(5)

    PHONE = phone_server.accept()[0]
    print("Phone Connected")

    # EV3.connect((EV3_IP, EV3_PORT))
    # EV3_EXIT.connect((EV3_IP, EV3_EXIT_PORT))
    # print("EV3 connected")

    run()


if __name__ == "__main__":
    main()
