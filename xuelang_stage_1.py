# -*- coding: utf-8 -*-

import  cv2
import  os
from glob import  glob

import scipy.misc as misc
import xml.etree.ElementTree as ET


dataset_dir = "/home/robin/文档/dataset/xuelang/data"

def read_image_dir(dir , prefix= ""):
    file_list = glob("{0}/*/*.jpg".format(dir))
    x =

    for index, file in enumerate(file_list):
        img = cv2.imread(file, cv2.IMREAD_COLOR)
        if img is not  None:
           img = misc.imresize(img , (512,512))
        xml_path = file.replace(".jpg",".xml")
        cat = get_text(xml_path)
        assert(cat != None)
        print(img.shape , cat)

def get_text(path):
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