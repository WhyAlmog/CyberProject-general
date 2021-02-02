from Utils import *
import socket

TOUCH_SENSOR = "1"

TRAPDOOR_MOTOR = "D"
TRAPDOOR_SPEED = str(1440)  # deg/s
TRAPDOOR_ANGLE = str(int(360 * 7.7))


#! A - right motor, positive degrees cause the rope to go LOOSE
#! B - left motor,  positive degrees cause the rope to go TAUT
TUBE_MOTOR_SPEED = str(360)
TUBE_MOTOR_ROTATIONS = str(int(360 * 4.5))


# ? 0/1/2 - left/middle/right
TUBE_POSITION = 1


def wait_touch_sensor_clicked(conn: socket) -> str:
    send(conn, "sensor_touch_wait_until_clicked " + TOUCH_SENSOR)
    return receive_string(conn)


def open_trapdoor(conn: socket) -> str:
    send(conn, "motor_run_angle " + TRAPDOOR_MOTOR +
         " " + TRAPDOOR_SPEED + " -" + TRAPDOOR_ANGLE)  # ! NOTICE THE - BEFORE THE ANGLE
    return receive_string(conn)


def close_trapdoor(conn: socket) -> str:
    send(conn, "motor_run_angle " + TRAPDOOR_MOTOR +
         " " + TRAPDOOR_SPEED + " " + TRAPDOOR_ANGLE)
    return receive_string(conn)


def tube_right_position(conn: socket):
    global TUBE_POSITION
    if TUBE_POSITION == 1:
        send(conn, "motor_run_angle A -" + TUBE_MOTOR_SPEED + " " +
             TUBE_MOTOR_ROTATIONS + " HOLD False")
        receive_string(conn)

        send(conn, "motor_run_angle B -" +
             TUBE_MOTOR_SPEED + " " + TUBE_MOTOR_ROTATIONS)
        receive_string(conn)

        TUBE_POSITION = 2


def tube_middle_position(conn: socket):
    global TUBE_POSITION
    if TUBE_POSITION == 2:
        send(conn, "motor_run_angle B " + TUBE_MOTOR_SPEED + " " +
             TUBE_MOTOR_ROTATIONS + " HOLD False")
        receive_string(conn)

        send(conn, "motor_run_angle A " + TUBE_MOTOR_SPEED + " " +
             TUBE_MOTOR_ROTATIONS)
        receive_string(conn)

        TUBE_POSITION = 1

    elif TUBE_POSITION == 0:
        send(conn, "motor_run_angle A -" + TUBE_MOTOR_SPEED + " " +
             TUBE_MOTOR_ROTATIONS + " HOLD False")
        receive_string(conn)

        send(conn, "motor_run_angle B -" + TUBE_MOTOR_SPEED + " " +
             TUBE_MOTOR_ROTATIONS)
        receive_string(conn)

        TUBE_POSITION = 1


def tube_left_position(conn: socket):
    global TUBE_POSITION
    if TUBE_POSITION == 1:
        send(conn, "motor_run_angle B " +
             TUBE_MOTOR_SPEED + " " + TUBE_MOTOR_ROTATIONS + " HOLD False")
        receive_string(conn)

        send(conn, "motor_run_angle A " + TUBE_MOTOR_SPEED + " " +
             TUBE_MOTOR_ROTATIONS)
        receive_string(conn)

        TUBE_POSITION = 0
