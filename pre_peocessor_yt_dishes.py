# -*- coding: utf-8 -*-、
# author : Robin
# 筛选生产环境数据

import  cv2
import  os
from glob import  glob

import scipy.misc as misc

GLCM_DIR = "patch_32"
patch_size = (32 ,32)
#git_mean_features = np.zeros([4,12],dtype=np.float32)

#Work_Dir = "./dish_rawdata/judge_false/"

date = "0330"


def clip_image(img , num  ):

    patch_dir = "./temp/{0}/{2}/{1}/".format( GLCM_DIR,num , date)
    ensure_directory_exist(patch_dir)

    mod_1 = img.shape[1] % patch_size[1]#最大列
    mod_2 = img.shape[0] % patch_size[0]#最大行
    for row in range(0 , img.shape[0] - mod_2 , patch_size[1]):
        for col in range(0 , img.shape[1] - mod_1, patch_size[0]):
                patch = img[ row: row + patch_size[1] ,
                             col: col + patch_size[0] ]
                cv2.imwrite(patch_dir + "{2}_{0}_{1}.jpg".format(row,col,num) , patch)

def ensure_directory_exist( directory_name):
    exist_bool = os.path.isdir(directory_name)
    if not exist_bool:
        # os.mkdir(directory_name)
        os.makedirs(directory_name)
    return

def clip_image_dir(dir , prefix):
    file_list = glob("{0}/*.*".format(dir))
    for index, file in enumerate(file_list):
        img = cv2.imread(file, cv2.IMREAD_COLOR)
        if img is not  None:
           img = misc.imresize(img , (512,512))
           clip_image(img, prefix + "_" + str(index))

clip_image_dir("../collect/pro/{0}/judge_false/".format(date) , "{0}_test_false".format(date))
clip_image_dir("../collect/pro/{0}/judge_true/".format(date) , "{0}_test_true".format(date))


# def clip_image_arr(arr):
#     for num in arr:
#         name = Work_Dir + "{0}.jpg".format(num)  #
#         img = cv2.imread(name, cv2.IMREAD_COLOR)
#         if img is not  None:
#            img = misc.imresize(img , (512,512))
#            clip_image(img, num)




#a = []
# for x in range(17):
#     a.append("true_false_{0}".format(x))
#
#
# for x in range(37):
#     a.append("true_true_{0}".format(x))
#
# for x in range(29):
#     a.append("false_true_{0}".format(x))
# for x in range(29):
#      a.append("judge_false_{0}".format(x))
#
#
# clip_image_arr(a)
