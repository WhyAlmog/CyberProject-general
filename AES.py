from Crypto.Cipher import AES
import base64


class AESEncrypt:
    def __init__(self, key: bytes, iv: bytes):
        """creates an AES encryption/decryption object

        Args:
            key (bytes): 16 bytes seed to use for key generation
            iv (bytes): 16 bytes seed to use for iv generation
        """
        self.key = key
        self.iv = iv
        self.mode = AES.MODE_CBC

    def encrypt(self, text:str) -> str:
        """encrypt a string with the key and the iv given when creating this object

        Args:
            text (str): the text to encrypt

        Returns:
            str: encrypted text, in base 64
        """
        cryptor = AES.new(self.key, self.mode, self.iv)
        length = AES.block_size
        text_pad = self.padding(length, text)
        ciphertext = cryptor.encrypt(text_pad.encode("utf-8"))
        return str(base64.b64encode(ciphertext), encoding='utf-8')


    def padding(self, length:int, text:str) -> str:
        """adds padding to the given text string to match AES block size

        Args:
            length (int): AES block size
            text (str): the text to pad

        Returns:
            str: padded text
        """
        count = len(text.encode('utf-8'))
        if count % length != 0:
            add = length - (count % length)
        else:
            add = 0
        return (text + ('\0' * add))

    def decrypt(self, text:bytes) -> bytes:
        """decrypts a given text using the key and iv given when creating this object

        Args:
            text (bytes): encrypted data, in base 64

        Returns:
            bytes: decrypted data
        """
        base_text = base64.b64decode(text)
        cryptor = AES.new(self.key, self.mode, self.iv)
        return cryptor.decrypt(base_text)