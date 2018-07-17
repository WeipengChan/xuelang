# -*- coding: utf-8 -*-

import  cv2
import  os
from glob import  glob

import scipy.misc as misc
import xml.etree.ElementTree as ET
import numpy as np

dataset_dir = "/home/robin/文档/dataset/xuelang/data"#这是在家里的linux系统路径，别删～～
dataset_dir = "/home/ubuntu/data/resource/dataset/xuelang/data"

def read_image_dir(dir , prefix= ""):
    file_list = glob("{0}/*/*.jpg".format(dir))
    x = np.zeros((len(file_list),512,512,3),dtype=np.int8)
    label = []
    for index, file in enumerate(file_list):
        img = cv2.imread(file, cv2.IMREAD_COLOR)
        if img is not  None:
           img = misc.imresize(img , (512,512))
        xml_path = file.replace(".jpg",".xml")
        cat = get_text(xml_path)
        assert(cat != None)
        print(img.shape , cat)
        x[index] = img
        label.append( cat)
    np.save("./x.npy",np.array(x))

    map = np.unique(np.array(label))
    #print(np.unique(np.array(y)))
    map_dict = {}
    for index,l in enumerate(map):
        map_dict[l] = index
    print(map_dict)

    y = np.zeros((len(file_list),),dtype=np.int8)
    for index,l in enumerate(label):
        print(l)
        y[index] = map_dict[l]
    y = np.array(y)

    print(x.shape , y.shape)

    np.save("./y.npy",np.array(y))

def get_text(path):
    if not os.path.isfile(path):
        return "正常"

    tree = ET.ElementTree(file= path)
    # print(tree)
    root = tree.getroot()
    #print(root,root.tag, root.attrib)
    for child_of_root in root:
        #print(child_of_root.tag, child_of_root.attrib)
        if child_of_root.tag == "object":
            for elem in child_of_root:
               # print("--",elem.tag, elem.text)
                if elem.tag == "name":
                    #print("----",elem.text)
                    return elem.text

read_image_dir(dataset_dir)