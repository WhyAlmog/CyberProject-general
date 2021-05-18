from AES import AESEncrypt
import hashlib
import socket
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5
from base64 import b64decode
from base64 import b64encode

READ_SIZE = 4096


def send(conn: socket, data: str, aes: AESEncrypt) -> None:
    data = aes.encrypt(data)
    send_no_aes(conn, data)


def send_no_aes(conn: socket, data: str) -> None:
    data = data.encode()

    conn.send(len(data).to_bytes(4, "big"))
    conn.send(data)


def receive(conn: socket, aes: AESEncrypt) -> bytes:
    return aes.decrypt(receive_no_aes(conn))


def receive_no_aes(conn: socket) -> bytes:
    """Receives data according to the 4 bytes header protocol, see send function"""
    length = int.from_bytes(conn.recv(4), "big")

    remaining = length
    data = b''
    while remaining != 0:
        if remaining < READ_SIZE:
            data += conn.recv(remaining)
        else:
            data += conn.recv(READ_SIZE)

        remaining = length - len(data)

    return data


def receive_string(conn: socket, aes: AESEncrypt) -> str:
    return receive(conn, aes).decode("utf-8")


def receive_string_no_aes(conn: socket) -> str:
    return receive_no_aes(conn).decode("utf-8")


def rsa_encrypt(s, public_key) -> str:
    cipher = PKCS1_v1_5.new(public_key)
    ciphertext = b64encode(cipher.encrypt(bytes(s, "utf-8")))
    return ciphertext


def rsa_decrypt(s, private_key) -> str:
    cipher = PKCS1_v1_5.new(private_key)
    plaintext = cipher.decrypt(b64decode(s), "Error while decrypting")
    return plaintext


def establish_connection(conn: socket) -> AESEncrypt:
    key = RSA.generate(2048)

    public_key = key.publickey().exportKey('PEM').decode()
    public_key = public_key.replace("-----BEGIN PUBLIC KEY-----", "")
    public_key = public_key.replace("-----END PUBLIC KEY-----", "")
    public_key = public_key.replace("\n", "")
    send_no_aes(conn, public_key)

    key_seed = rsa_decrypt(receive_string_no_aes(conn), key)
    iv_seed = rsa_decrypt(receive_string_no_aes(conn), key)

    print(key_seed)

    return AESEncrypt(key=key_seed, iv=iv_seed)
