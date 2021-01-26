import socket
from EV3Commands import *

EV3 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
EV3_IP = "192.168.137.3"
EV3_PORT = 8070

EV3_EXIT = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
EV3_EXIT_PORT = 8071

def main():
    EV3.connect((EV3_IP, EV3_PORT))
    EV3_EXIT.connect((EV3_IP, EV3_EXIT_PORT))

    send(EV3, "motor_run_angle D 720 -2880")
    receive_string(EV3)
    send(EV3, "sensor_touch_wait_until_clicked 1")
    receive_string(EV3)
    send(EV3, "motor_run_angle D 720 2880")
    receive_string(EV3)


if __name__ == "__main__":
    main()
