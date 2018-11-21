#!/usr/bin/env python
import os
import numpy as np
from matplotlib import pyplot as plt
import os
import matplotlib.image as mpimg
import cv2
import torch
import sys

N_IMAGES = 23708
IMG_SIZE = 256
IMG_PATH = 'images_%i_%i.pth' % (IMG_SIZE, IMG_SIZE)
ATTR_PATH = 'attributes.pth'


def get_counts():
    file_names = os.listdir('UTKFace')

    age = np.array([int(file_names[i].split('_')[0]) for i in range(len(file_names))])
    age_count = np.zeros(117)
    for i in range(len(age)):
        age_count[age[i]] += 1

    gender = np.array([int(file_names[i].split('_')[1]) for i in range(len(file_names))])
    gender_count = np.zeros(2)
    for i in range(len(gender)):
        gender_count[gender[i]] += 1

    race = np.array([int(file_names[i].split('_')[2]) for i in range(len(file_names))])
    race_count = np.zeros(5)
    for i in range(len(race)):
        race_count[race[i]] += 1
    return file_names, age, gender, race


file_names, age, gender, race = get_counts()


def preprocess_images():

    if os.path.isfile(IMG_PATH):
        print("%s exists, nothing to do." % IMG_PATH)
        return

    print("Reading images from UTKFace/ ...")
    raw_images = []
    for i in range(N_IMAGES):
        if i % 2000 == 0:
            print(i)
        raw_images.append(mpimg.imread("UTKFace/%s" % file_names[i]))

    if len(raw_images) != N_IMAGES:
        raise Exception("Found %i images. Expected %i" % (len(raw_images), N_IMAGES))

    print("Resizing images ...")
    all_images = []
    for i, image in enumerate(raw_images):
        if i % 2000 == 0:
            print(i)
        assert image.shape == (200, 200, 3)
        if IMG_SIZE < 200:
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        elif IMG_SIZE > 200:
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
        assert image.shape == (IMG_SIZE, IMG_SIZE, 3)
        all_images.append(image)

    print("Transposing the vectors...")
    data = np.concatenate([img.transpose((2, 0, 1))[None] for img in all_images], 0)
    data = torch.from_numpy(data)
    assert data.size() == (N_IMAGES, 3, IMG_SIZE, IMG_SIZE)

    print("Saving images to %s ..." % IMG_PATH)
    torch.save(data[:20000].clone(), 'images_%i_%i_20000.pth' % (IMG_SIZE, IMG_SIZE))
    torch.save(data, IMG_PATH)


def preprocess_attributes(attr_names, attr_list):

    if os.path.isfile(ATTR_PATH):
        print("%s exists, nothing to do." % ATTR_PATH)
        return

    attr_lines = list(zip(*attr_list))
    assert len(attr_lines) == N_IMAGES

    attributes = {k: np.zeros(N_IMAGES, dtype=np.uint8) for k in attr_names}

    for i, attr in enumerate(attr_lines):
        image_id = i + 1
        assert attr[0] == int(file_names[i].split('_')[1])
        assert attr[1] == int(file_names[i].split('_')[2])
        assert all(x in [0, 1] for x in [attr[0]])
        assert all(x in [0, 1, 2, 3, 4, 5] for x in [attr[1]])
        for j, value in enumerate(attr):
            attributes[attr_names[j]][i] = value

    print("Saving attributes to %s ..." % ATTR_PATH)
    torch.save(attributes, ATTR_PATH)


preprocess_images()

a_n = ['Gender', 'Race']
a_l = [gender, race]
preprocess_attributes(a_n, a_l)
