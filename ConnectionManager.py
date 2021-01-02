import socket
import threading
import time

from EV3Commands import WAIT_TOUCH_SENSOR_CLICKED # type:ignore
from EV3Commands import TEST_MOVE # type:ignore


FOLDER = "C:\\Users\\almog\\Desktop\\test"

EV3_IP = "192.168.137.3"
EV3_PORT = 8070
EV3_EXIT_PORT = 8071
PHONE_SERVER_PORT = 9768

EV3 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
EV3_EXIT = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
PHONE = None

RUNNING = True


def send(conn:socket, data:str):
    """Sends data as bytes with a 4 byte header acting as the message's length (big endian)"""
    data = data.encode()
    conn.send(len(data).to_bytes(4, "big"))
    conn.send(data)


def receive(conn:socket):
    """Receives data according to the 4 bytes header protocol, see send function"""
    length = int.from_bytes(conn.recv(4), "big")
    return conn.recv(length)


def receive_string(conn:socket):
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
            with open(FOLDER + "/Time" + str(time.time()) + ".jpg", 'wb') as f:
                f.write(receive(PHONE))
            send(EV3, TEST_MOVE)
            receive_string(EV3)


def exit(phone_server:socket):
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
