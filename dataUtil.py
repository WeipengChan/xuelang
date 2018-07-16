# -*- coding: utf-8 -*-
# author : Robin

import sys

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import cv2
import os.path
from glob import glob
from PIL import Image

#from imgaug import augmenters as iaa

ImgWidth = 128
ImgHeight = 128
Channel = 3

# import matplotlib #this class is used for ipynb, so I won't disabled matplolib
# matplotlib.use('Agg')


def ensure_directory_exist( directory_name):
    exist_bool = os.path.isdir(directory_name)
    if not exist_bool:
        # os.mkdir(directory_name)
        os.makedirs(directory_name)
    return
#####################################################################################################################
#label loading
# read csv
def readClassCsv():
    #path = os.path.abspath(os.path.join("./", os.pardir))
    csv = pd.read_csv("./signnames.csv",encoding="utf-8")
    csv.set_index("ClassId")
    return csv
    # print(sign_names.head(n=10))

def readEstimateClassCsv():
    #path = os.path.abspath(os.path.join("./", os.pardir))
    csv = pd.read_csv("./estimate_names.csv",encoding="utf-8")
    csv.set_index("ClassId")
    return csv


# input y
def map_img_id_to_lbl(lbs_ids, lbs_names):
    """
    Utility function to group images by label
    """
    print(lbs_ids)
    print(lbs_names.shape)

    arr_map = []
    for i in range(0, lbs_ids.shape[0]):
        label_id = lbs_ids[i]
        # print("name:" +  str(lbs_names["ClassId"]))
        label_name = lbs_names[lbs_names["ClassId"] == label_id]["SignName"].values[0]
        #print(label_name)
        arr_map.append({"img_id": i, "label_id": label_id, "label_name": label_name})

    return pd.DataFrame(arr_map)

#####################################################################################################################
# draw distribution
def draw_img_id_to_lb_count(img_id_to_lb):
    """
    Returns a pivot table table indexed by label id and label name, where the aggregate function is count
    """
    table = pd.pivot_table(img_id_to_lb, index=["label_id", "label_name"], values=["img_id"], aggfunc='count')
    table.plot(kind='bar', figsize=(10, 7))


def show_image_list(img_list, img_labels, title, cols=2, fig_size=(1, 1), show_ticks=True):
    """
    Utility function to show us a list of traffic sign images
    """
    img_count = len(img_list)
    rows = img_count // cols
    cmap = None

    fig, axes = plt.subplots(rows, cols, figsize=fig_size)

    for i in range(0, img_count):
        img_name = img_labels[i]
        img = img_list[i]
        if len(img.shape) < 3 or img.shape[-1] < 3:
            cmap = "gray"
            img = np.reshape(img, (img.shape[0], img.shape[1]))

        if not show_ticks:
            axes[i].axis("off")

        axes[i].imshow(img, cmap=cmap)

    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.6)
    fig.tight_layout()
    plt.show()

    return


def show_random_dataset_images(group_label, imgs, to_show=5):
    """
    This function takes a DataFrame of items group by labels as well as a set of images and randomly selects to_show images to display
    """
    for (lid, lbl), group in group_label:
        # print("[{0}] : {1}".format(lid, lbl))
        rand_idx = np.random.randint(0, high=group['img_id'].size, size=to_show, dtype='int')
        selected_rows = group.iloc[rand_idx]
        # print(selected_rows)

        selected_img = list(map(lambda img_id: imgs[img_id], selected_rows['img_id']))
        selected_labels = list(map(lambda label_id: label_id, selected_rows['label_id']))
        show_image_list(selected_img, selected_labels, "{0}: {1}".format(lid, lbl), cols=to_show, fig_size=(7, 7),
                        show_ticks=True)


#####################################################################################################################




#################################################################################################################

def matrixToImage(data):
    data = data * 255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im


#################################################################################################################
#data normal and gray scale
def normalise_images(imgs, dist):
    """
    Nornalise the supplied images from data in dist
    """
    std = np.std(dist)
    #std = 128
    mean = np.mean(dist)
    #mean = 128
    return (imgs - mean) / std



def to_grayscale(img):
    """
    Converts an image in RGB format to grayscale
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#################################################################################################################
#data arugment
def augment_imgs(imgs, p):
    """
    Performs a set of augmentations with with a probability p
    """
    augs = iaa.SomeOf((1, 2),
                      [
                          iaa.Crop(px=(0, 4)),  # crop images from each side by 0 to 4px (randomly chosen)
                          iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
                          iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
                          iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to +45 degrees)
                          iaa.Affine(shear=(-10, 10))  # shear by -10 to +10 degrees
                      ])

    seq = iaa.Sequential([iaa.Sometimes(p, augs)])

    return seq.augment_images(imgs)


def augment_imgs_until_n(imgs, n, p):
    """
    Takes care of augmenting images with a probability p, until n augmentations have been created
    """

    i = 0
    aug_imgs = []
    while i < n:
        augs = augment_imgs(imgs, p)
        i += len(augs)
        aug_imgs = augs if len(aug_imgs) == 0 else np.concatenate((aug_imgs, augs))

    return aug_imgs[0: n]
#convience for augment but I has not tested
def augment_imgs_convience(x,y, group_by_label):
    # This loop augments images per label group
    X_train_augs = x
    y_train_augs = y
    for (lid, lbl), group in group_by_label:
        # print("[{0}] : {1}".format(lid, lbl))
        group_count = group['img_id'].size
        idx = group['img_id'].values
        imgs = x[idx]

        # Take a different population of the subset depending on how many images we have already
        # and vary the number of augmentations depending on size of label group
        pt_spacing = 1.0
        p = 1.0

        n = group_count * 0.1

        if group_count > 500 and group_count < 1000:
            pt_spacing = 3.0
        elif group_count >= 1000 and group_count < 2000:
            pt_spacing = 10.0
        elif group_count >= 2000:
            pt_spacing = 20.0

        n = int(n)

        space_interval = int(group_count / pt_spacing)

        rand_idx = np.linspace(0, group_count, num=space_interval, endpoint=False, dtype='int')

        selected_rows = group.iloc[rand_idx]
        selected_img = np.array(list(map(lambda img_id: x[img_id], selected_rows['img_id'])))

        augs = augment_imgs_until_n(selected_img, n, p)
        X_train_augs = np.concatenate((X_train_augs, augs))
        y_train_augs = np.concatenate((y_train_augs, np.repeat(lid, n)))

    print("New Augmented arrays shape: {0} and {1}".format(X_train_augs.shape, y_train_augs.shape))
    return X_train_augs, y_train_augs


#################################################################################################################
#extrac sub set
def random_sample_set(grouped_imgs_by_label, imgs, labels, pct=0.4):
    """
    Creates a sample set containing pct elements of the original grouped dataset
    """
    X_sample = []
    y_sample = []

    for (lid, lbl), group in grouped_imgs_by_label:
        group_size = group['img_id'].size
        img_count_to_copy = int(group_size * pct)
        rand_idx = np.random.randint(0, high=group_size, size=img_count_to_copy, dtype='int')

        selected_img_ids = group.iloc[rand_idx]['img_id'].values
        selected_imgs = imgs[selected_img_ids]
        selected_labels = labels[selected_img_ids]
        X_sample = selected_imgs if len(X_sample) == 0 else np.concatenate((selected_imgs, X_sample), axis=0)
        y_sample = selected_labels if len(y_sample) == 0 else np.concatenate((selected_labels, y_sample), axis=0)

    return (X_sample, y_sample)
#################################################################################################################
# imporve contrast if grayscale is too dark
def improveContras(grayscale):
    clahe = cv2.createCLAHE(tileGridSize=(2, 2), clipLimit=15.0)
    grayscale_equalized = np.asarray(list(map(lambda img: clahe.apply(np.reshape(img, (ImgWidth, ImgHeight))), grayscale)))
    return np.reshape(grayscale_equalized, (grayscale_equalized.shape[0], ImgWidth, ImgWidth, 1))

#################################################################################################################



#################################################################################################################


# if __name__ == '__main__':
#     pattern = "../datasets/000000/*.jpg"
#     X_train = get_imgs_by_file_pattern(pattern)
#     X_train.shape

     # csv = readClassCsv()
     # y = np.ndarray([4,],dtype=np.float32)
     # y[0] = 0
     # y[1] = 2
     # y[2] = 3
     # y[3] = 4
     # dataFrame = map_img_id_to_lbl(y , csv)
     # draw_img_id_to_lb_count(dataFrame)
