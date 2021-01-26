import socket

def send(conn: socket, data: str):
    """Sends data as bytes with a 4 byte header acting as the message's length (big endian)"""
    data = data.encode()
    conn.send(len(data).to_bytes(4, "big"))
    conn.send(data)


def receive(conn: socket):
    """Receives data according to the 4 bytes header protocol, see send function"""
    length = int.from_bytes(conn.recv(4), "big")
    return conn.recv(length)


def receive_string(conn: socket):
    """Receives data according to the 4 bytes header protocol, see send function"""
    length = int.from_bytes(conn.recv(4), "big")
    return conn.recv(length).decode()