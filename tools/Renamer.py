import os

FOLDER = "D:\\Datasets\\CyberProject\\paper\\"

starting_name = 0

for filename in os.listdir(FOLDER):
    os.rename(FOLDER + filename, FOLDER + str(starting_name) + ".jpg")
    starting_name += 1
