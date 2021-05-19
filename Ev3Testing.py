import socket
from EV3Commands import *

EV3 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
EV3_IP = "192.168.137.3"
EV3_PORT = 8070

EV3_EXIT = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
EV3_EXIT_PORT = 8071


def main():
    """used to test the EV3 without a need for a phone
    """
    EV3.connect((EV3_IP, EV3_PORT))
    EV3_EXIT.connect((EV3_IP, EV3_EXIT_PORT))

    tube_right_position(EV3)
    wait_touch_sensor_clicked(EV3)
    tube_middle_position(EV3)
    wait_touch_sensor_clicked(EV3)
    tube_left_position(EV3)
    wait_touch_sensor_clicked(EV3)
    tube_middle_position(EV3)


if __name__ == "__main__":
    main()
