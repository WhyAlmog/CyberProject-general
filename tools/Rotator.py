import cv2
import os

FOLDER = "D:\\Datasets\\CyberProject\\tin\\"

new_name = sorted(int(name.split(".")[0]) for name in os.listdir(FOLDER))[-1]


for filename in os.listdir(FOLDER):
    img = cv2.imread(FOLDER + filename, cv2.IMREAD_UNCHANGED)
    img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(FOLDER + str(new_name) + ".jpg", img)
    new_name += 1
    img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(FOLDER + str(new_name) + ".jpg", img)
    new_name += 1
    img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(FOLDER + str(new_name) + ".jpg", img)
    new_name += 1
