from Utils import *
import socket

# touch sensor port
TOUCH_SENSOR = "1"

# trapdoor motor (small motor) port
TRAPDOOR_MOTOR = "D"
TRAPDOOR_SPEED = str(1440)  # deg/s
TRAPDOOR_ANGLE = str(int(360 * 7.7))  # empirically tested to be this number


#! A - right motor, positive degrees cause the rope to go LOOSE
#! B - left motor,  positive degrees cause the rope to go TAUT
TUBE_MOTOR_SPEED = str(int(360))  # deg/s
TUBE_MOTOR_ROTATIONS = str(int(360 * 4.5))  # field tested to be this number


#! 0/1/2 - left/middle/right
TUBE_POSITION = 1


def wait_touch_sensor_clicked(conn: socket) -> str:
    """waits for the touch sensor to be clicked (unpressed -> pressed -> unpressed)

    Args:
        conn (socket): connection to the EV3

    Returns:
        str: success if successful, error message if not
    """
    send(conn, "sensor_touch_wait_until_clicked " + TOUCH_SENSOR)
    return receive_string(conn)


def open_trapdoor(conn: socket) -> str:
    """opens the trapdoor

    Args:
        conn (socket): connection to the EV3

    Returns:
        str: success if successful, error message if not
    """
    send(conn, "motor_run_angle " + TRAPDOOR_MOTOR +
         " " + TRAPDOOR_SPEED + " -" + TRAPDOOR_ANGLE)  # ! NOTICE THE - BEFORE THE ANGLE
    return receive_string(conn)


def close_trapdoor(conn: socket) -> str:
    """closes the trapdoor

    Args:
        conn (socket): connection to the EV3

    Returns:
        str: success if successful, error message if not
    """
    send(conn, "motor_run_angle " + TRAPDOOR_MOTOR +
         " " + TRAPDOOR_SPEED + " " + TRAPDOOR_ANGLE)
    return receive_string(conn)


def tube_right_position(conn: socket):
    """move the trapdoor to the right position    

    Args:
       conn (socket): connection to the EV3
    """
    global TUBE_POSITION
    if TUBE_POSITION == 1:
        send(conn, "motor_run_angle A -" + TUBE_MOTOR_SPEED + " " +
             TUBE_MOTOR_ROTATIONS + " HOLD False")
        receive(conn)

        send(conn, "motor_run_angle B -" +
             TUBE_MOTOR_SPEED + " " + TUBE_MOTOR_ROTATIONS)
        receive(conn)

        TUBE_POSITION = 2


def tube_middle_position(conn: socket):
    """move the trapdoor to the middle position    

    Args:
       conn (socket): connection to the EV3
    """
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
    """move the trapdoor to the left position    

    Args:
       conn (socket): connection to the EV3
    """
    global TUBE_POSITION
    if TUBE_POSITION == 1:
        send(conn, "motor_run_angle B " +
             TUBE_MOTOR_SPEED + " " + TUBE_MOTOR_ROTATIONS + " HOLD False")
        receive_string(conn)

        send(conn, "motor_run_angle A " + TUBE_MOTOR_SPEED + " " +
             TUBE_MOTOR_ROTATIONS)
        receive_string(conn)

        TUBE_POSITION = 0
