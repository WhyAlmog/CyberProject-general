import socket

PHONE = None
PHONE_SERVER_PORT = 9768


def main():
    global PHONE

    phone_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    phone_server.bind(('0.0.0.0', PHONE_SERVER_PORT))
    phone_server.listen(5)

    PHONE = phone_server.accept()[0]
    print("Phone Connected")


if __name__ == "__main__":
    main()
