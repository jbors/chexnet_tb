"""
Automatically generate train/val/test sets from labeled images.
Anything ending in _1.png is a positive example, negative otherwise.
"""


import os
from random import randint

SOURCE_DIR = "./XRAY_images/images/"
TARGET_DIR = "./XRAY_images/labels/"

train_file = open(TARGET_DIR + 'train_list.txt', "w")
test_file = open(TARGET_DIR + 'test_list.txt', "w")
val_file = open(TARGET_DIR + 'val_list.txt', "w")

for filename in os.listdir(SOURCE_DIR):
    if filename.endswith(".png"):
        line = filename + " 0 1\n"
        if filename.endswith("_1.png"):
            line = filename + " 1 0\n"
        cutoff = randint(0, 100)
        if cutoff < 80:
            train_file.write(line)
        elif cutoff < 90:
            test_file.write(line)
        else:
            val_file.write(line)

train_file.close()
test_file.close()
val_file.close()
