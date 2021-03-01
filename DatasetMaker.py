import cv2
import os
import numpy as np
import random
from skimage import io


FOLDERS = ["C:\\Users\\almog\\OneDrive\\VSCodeWorkspace\\CyberProject\\CyberProject-data\\datasetProcessed\\test\\paper\\",
           "C:\\Users\\almog\\OneDrive\\VSCodeWorkspace\\CyberProject\\CyberProject-data\\datasetProcessed\\test\\tin\\",
           "C:\\Users\\almog\\OneDrive\\VSCodeWorkspace\\CyberProject\\CyberProject-data\\datasetProcessed\\test\\plastic\\",
           "C:\\Users\\almog\\OneDrive\\VSCodeWorkspace\\CyberProject\\CyberProject-data\\datasetProcessed\\train\\paper\\",
           "C:\\Users\\almog\\OneDrive\\VSCodeWorkspace\\CyberProject\\CyberProject-data\\datasetProcessed\\train\\tin\\",
           "C:\\Users\\almog\\OneDrive\\VSCodeWorkspace\\CyberProject\\CyberProject-data\\datasetProcessed\\train\\plastic\\"]


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


def noise(folder):
    new_name = sorted(int(name.split(".")[0])
                      for name in os.listdir(folder))[-1]

    for filename in os.listdir(folder):
        image = cv2.imread(folder + filename, cv2.IMREAD_UNCHANGED)
        noise_img = sp_noise(image, 0.005)
        cv2.imwrite(folder + str(new_name) + ".jpg", noise_img)
        new_name += 1


def rotate(folder):
    new_name = sorted(int(name.split(".")[0])
                      for name in os.listdir(folder))[-1]

    for filename in os.listdir(folder):
        img = cv2.imread(folder + filename, cv2.IMREAD_UNCHANGED)
        img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(folder + str(new_name) + ".jpg", img)
        new_name += 1
        img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(folder + str(new_name) + ".jpg", img)
        new_name += 1
        img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(folder + str(new_name) + ".jpg", img)
        new_name += 1


def rename(folder):
    starting_name = 100000

    for filename in os.listdir(folder):
        os.rename(folder + filename, folder + str(starting_name) + ".jpg")
        starting_name += 1

    starting_name = 0

    for filename in os.listdir(folder):
        os.rename(folder + filename, folder + str(starting_name) + ".jpg")
        starting_name += 1


def verify(folder):
    files_to_remove = []

    for filename in os.listdir(folder):
        try:
            _ = io.imread(folder + filename)
        except Exception as e:
            print(filename)
            files_to_remove.append(filename)

    for filename in files_to_remove:
        os.remove(folder + filename)


for folder in FOLDERS:
    print(folder)
    verify(folder)
    rename(folder)
    rotate(folder)
    # noise(folder)

print("finished")
