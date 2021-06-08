from AES import AESEncrypt
import socket
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5
from base64 import b64decode
from base64 import b64encode

READ_SIZE = 4096


def send(conn: socket, data: str, aes: AESEncrypt = None) -> None:
    """send data through a socket, and encrypts it using the symmetric AES encryption algorithm.
    this will send the length of the data in 4 bytes (big endian) before sending the actual data.

    Args:
        conn (socket): the socket to send data through
        data (str): the data to send
        aes (AESEncrypt, optional): aes object to encrypt with. Defaults to None, where the data won't be encrypted.
    """ 
    if aes is not None:
        data = aes.encrypt(data)

    data = data.encode("utf-8")

    conn.send(len(data).to_bytes(4, "big"))
    conn.send(data)


def receive(conn: socket, aes: AESEncrypt = None) -> bytes:
    """receives data from a socket, and decrypts it using the symmetric AES encryption algorithm.
    follows the same 4-byte-length protocol described in send function

    Args:
        conn (socket): the socket to read the data from
        aes (AESEncrypt, optional): aes object to decrypt with. Defaults to None, where the data won't be decrypted.

    Returns:
        bytes: data read from the socket, decrypted
    """
    length = int.from_bytes(conn.recv(4), "big")

    remaining = length
    data = b''
    while remaining != 0:
        if remaining < READ_SIZE:
            data += conn.recv(remaining)
        else:
            data += conn.recv(READ_SIZE)

        remaining = length - len(data)

    if aes is not None:
        data = aes.decrypt(data)

    return data


def receive_string(conn: socket, aes: AESEncrypt = None) -> str:
    """converts the bytes read from the receive function to a utf-8 string

    Args:
        conn (socket): socket to receive the data from
        aes (AESEncrypt, optional): aes object to decrypt with. Defaults to None, where the data won't be decrypted.

    Returns:
        str: utf-8 formatted string received from the socket
    """
    return receive(conn, aes).decode("utf-8")


def rsa_encrypt(s:str, public_key:str) -> str:
    """encrypts data using the RSA a-symmetric encryption algorithm

    Args:
        s (str): string to encrypt
        public_key (str): public RSA key, in base 64

    Returns:
        str: the data encrypted using RSA, in base 64
    """
    cipher = PKCS1_v1_5.new(public_key)
    return b64encode(cipher.encrypt(bytes(s, "utf-8")))


def rsa_decrypt(s:str, private_key:str) -> bytes:
    """decrypts data using the RSA a-symmetric encryption algorithm

    Args:
        s (str): string to decrypt, in base 64
        private_key (str): private RSA key, in base 64

    Returns:
        str: decrypted string, in utf-8
    """
    cipher = PKCS1_v1_5.new(private_key)
    return cipher.decrypt(b64decode(s), "Error while decrypting")


def establish_connection(conn: socket) -> AESEncrypt:
    """establishes a secure connection between the server and the provided socket.
    the server will generate an RSA key pair and send the public key to the client.
    the client will generate AES key and iv seeds, encrypt them using the RSA key and send them to the server.
    the server will then decrypt the seeds and use them to create the AES key on his side.
    now both sides have the AES key and can communicate securely.

    Args:
        conn (socket): the socket to communicate through

    Returns:
        AESEncrypt: the aes object to later encrypt and decrypt with
    """
    key = RSA.generate(2048)

    public_key = key.publickey().exportKey('PEM').decode()
    public_key = public_key.replace("-----BEGIN PUBLIC KEY-----", "")
    public_key = public_key.replace("-----END PUBLIC KEY-----", "")
    public_key = public_key.replace("\n", "")
    send(conn, public_key)

    key_seed = rsa_decrypt(receive_string(conn), key)
    iv_seed = rsa_decrypt(receive_string(conn), key)

    return AESEncrypt(key=key_seed, iv=iv_seed)
