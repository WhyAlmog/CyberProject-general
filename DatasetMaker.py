from ConnectionManager import main
import typing
import cv2
import os
import numpy as np
import random
from skimage import io

# folders to process
FOLDERS = ["C:\\Users\\Manor\\OneDrive\\VSCodeWorkspace\\CyberProject\\CyberProject-data\\datasetProcessed\\test\\paper\\",
           "C:\\Users\\Manor\\OneDrive\\VSCodeWorkspace\\CyberProject\\CyberProject-data\\datasetProcessed\\test\\tin\\",
           "C:\\Users\\Manor\\OneDrive\\VSCodeWorkspace\\CyberProject\\CyberProject-data\\datasetProcessed\\test\\plastic\\",
           "C:\\Users\\Manor\\OneDrive\\VSCodeWorkspace\\CyberProject\\CyberProject-data\\datasetProcessed\\train\\paper\\",
           "C:\\Users\\Manor\\OneDrive\\VSCodeWorkspace\\CyberProject\\CyberProject-data\\datasetProcessed\\train\\tin\\",
           "C:\\Users\\Manor\\OneDrive\\VSCodeWorkspace\\CyberProject\\CyberProject-data\\datasetProcessed\\train\\plastic\\"]


def sp_noise(image: typing.Any, prob: float) -> typing.Any:
    """add random black and white noise

    Args:
        image (Any): cv2 image to add the noise to
        prob (float): noise probability 

    Returns:
        [Any]: image with the noise added
    """
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


def noise(folder: str):
    """apply noise on an entire folder

    Args:
        folder (str): folder path
    """
    new_name = sorted(int(name.split(".")[0])
                      for name in os.listdir(folder))[-1]

    for filename in os.listdir(folder):
        image = cv2.imread(folder + filename, cv2.IMREAD_UNCHANGED)
        noise_img = sp_noise(image, 0.005)
        cv2.imwrite(folder + str(new_name) + ".jpg", noise_img)
        new_name += 1


def rotate(folder: str):
    """create coppies of a picture rotated by 90,180,270 degrees,
    applies this process for every picture in the folder

    Args:
        folder (str): folder path
    """
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


def rename(folder: str):
    """serialize the names of the items in a folder (changes the names to: 0.jpg, 1.jpg, 2.jpg....)

    Args:
        folder (str): folder path
    """
    starting_name = 100000

    for filename in os.listdir(folder):
        os.rename(folder + filename, folder + str(starting_name) + ".jpg")
        starting_name += 1

    starting_name = 0

    for filename in os.listdir(folder):
        os.rename(folder + filename, folder + str(starting_name) + ".jpg")
        starting_name += 1


def verify(folder: str):
    """verifies the integrity of all of the images in a folder and deletes the truncated ones.

    Args:
        folder (str): folder path
    """
    files_to_remove = []

    for filename in os.listdir(folder):
        try:
            _ = io.imread(folder + filename)
        except Exception as e:
            print(filename)
            files_to_remove.append(filename)

    for filename in files_to_remove:
        os.remove(folder + filename)


def main():
    """start the program
    """
    for folder in FOLDERS:
        print(folder)
        verify(folder)
        rename(folder)
        rotate(folder)
        # noise(folder)
        # noise doesn't add much to the network accuracy while training so we can ignore it

    print("finished")


if __name__ == "__main__":
    main()
