from utils import *
import socket

TOUCH_SENSOR = "1"

TRAPDOOR_MOTOR = "D"
TRAPDOOR_SPEED = str(1440)  # deg/s
TRAPDOOR_ANGLE = str(int(360 * 7.7))


def wait_touch_sensor_clicked(conn: socket):
    send(conn, "sensor_touch_wait_until_clicked " + TOUCH_SENSOR)
    return receive_string(conn)


def open_trapdoor(conn: socket):
    send(conn, "motor_run_angle " + TRAPDOOR_MOTOR +
         " " + TRAPDOOR_SPEED + " -" + TRAPDOOR_ANGLE)  # NOTICE THE - BEFORE THE ANGLE!
    return receive_string(conn)


def close_trapdoor(conn: socket):
    send(conn, "motor_run_angle " + TRAPDOOR_MOTOR +
         " " + TRAPDOOR_SPEED + " " + TRAPDOOR_ANGLE)
    return receive_string(conn)
