from Crypto.Cipher import AES
import base64


class AESEncrypt:
    def __init__(self, key, iv):
        self.key = key.encode("utf-8")
        self.iv = iv.encode("utf-8")
        self.mode = AES.MODE_CBC

    def encrypt(self, text):
        cryptor = AES.new(self.key, self.mode, self.iv)
        length = AES.block_size
        text_pad = self.padding(length, text)
        ciphertext = cryptor.encrypt(text_pad.encode("utf-8"))
        cryptedStr = str(base64.b64encode(ciphertext), encoding='utf-8')
        return cryptedStr

    def padding(self, length, text):
        count = len(text.encode('utf-8'))
        if count % length != 0:
            add = length - (count % length)
        else:
            add = 0
        text1 = text + ('\0' * add)
        return text1

    def decrypt(self, text):
        base_text = base64.b64decode(text)
        cryptor = AES.new(self.key, self.mode, self.iv)
        plain_text = cryptor.decrypt(base_text)
        ne = plain_text.decode('utf-8').rstrip('\0')
        return ne


if __name__ == '__main__':
    aes_encrypt = AESEncrypt(key='keyskeyskeyskeys', iv="keyskeyskeyskeys")  # Initialize key and IV
    text = '123'
    sign_data = aes_encrypt.encrypt(text)
    print(sign_data)
    data = aes_encrypt.decrypt(sign_data)
    print(data)