import numpy as np
import random
import cv2
import os

FOLDER = "D:\\Datasets\\CyberProject\\plastic\\"


def sp_noise(image, prob):
    output = np.zeros(image.shape, np.uint8)
    thresh = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thresh:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


new_name = sorted(int(name.split(".")[0]) for name in os.listdir(FOLDER))[-1]

for filename in os.listdir(FOLDER):
    image = cv2.imread(FOLDER + filename, cv2.IMREAD_UNCHANGED)
    noise_img = sp_noise(image, 0.005)
    cv2.imwrite(FOLDER + str(new_name) + ".jpg", noise_img)
    new_name += 1
