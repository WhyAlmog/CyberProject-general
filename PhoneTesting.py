import socket
import time

from PIL import Image

from Utils import *

PHONE = None
PHONE_AES = None
PHONE_SERVER_PORT = 9768

IMAGE_FOLDER = "C:\\Users\\almog\\OneDrive\\VSCodeWorkspace\\CyberProject\\CyberProject-data\\images\\"


def run():
    while True:
        input()

        successful = False
        filepath = IMAGE_FOLDER + str(int(time.time())) + ".jpg"
        image = None
        while not successful:
            #try:
                send(PHONE, "TAKE PICTURE", PHONE_AES)
                with open(filepath, 'wb') as f:
                    f.write(receive(PHONE, PHONE_AES))

                image = Image.open(filepath)

                try:
                    image.load()
                    successful = True
                except Exception:
                    image.close()
            # except Exception:
            #     successful = False


def main():
    """test just the connection to the phone and picture taking, without a need for the EV3 to be connected
    """
    global PHONE
    global PHONE_AES

    phone_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    phone_server.bind(('0.0.0.0', PHONE_SERVER_PORT))
    phone_server.listen(5)

    PHONE = phone_server.accept()[0]
    print("Phone Connected")

    PHONE_AES = establish_connection(PHONE)
    print("encrypted")

    run()


if __name__ == "__main__":
    main()
