import os
from PIL import Image

IMAGE_FOLDER = "C:\\Users\\almog\\OneDrive\\VSCodeWorkspace\\CyberProject\\CyberProject-data\\images\\"


filepath = IMAGE_FOLDER + "temp.jpg"
image = Image.open(filepath)
try:
    image.load()
except Exception:
    image.close()
    os.remove(filepath)
