# -*- coding: utf-8 -*-、
# author : Robin
from __future__ import print_function
from __future__ import division
import scipy.io as sio
import tensorflow as tf
from ModelPipeline import  ModelExecutor , ModelConfig
import numpy as np
import cv2
from EdLeNet import EdLeNet
import dataUtil as du
import scipy.misc as misc
from glob import glob
import os
imgSize = (256,192)
#由于正样本少于负样本

JUDGE_RATE = 0.08
DIR_TRUE_SAMPLE = '/home/robin/文档/dataset/xuelang/data_binary/0'
DIR_FALSE_SAMPLE = '/home/robin/文档/dataset/xuelang/data_binary/1'
TrainingMode = True #开启训练模式

def read_dataset_pipeline( outputDataDir, isTrueLabel=True , isTestData = False):
    print("reading dataset", outputDataDir)

    fileList = glob(outputDataDir + "*")


    # 为了训练的严谨性,训练的样本从每张照片的切片中0和1各抽取一定量的样本进行
    train_data = None
    train_label = None
    valid_data = None
    valid_label = None

    test_data = None
    test_label = None

    lastIndex = 0
    # print fileList
    for file_name in fileList:
        t = get_imgs_labels_by_file_pattern(file_name , isTrueLabel = isTrueLabel ,isTestData = isTestData)

        if isTestData == True:
            test_data  = contcat(test_data , t[0])
            test_label = contcat(test_label , t[1])
        else:
            train_data = contcat(train_data, t[0] )
            train_label = contcat(train_label , t[1])
            valid_data  = contcat(valid_data , t[2])
            valid_label  = contcat(valid_label , t[3])

    if isTestData == True:
        return test_data ,test_label
    else:
        return train_data ,train_label ,valid_data ,valid_label

def contcat(arr , new_member):
    if arr is None:
       return  new_member
    else:
       return  np.concatenate((arr , new_member))




# 2017年12月04日16:51:33 prepare_method,如果不为None则使用prepare_method,进行增量处理
def get_imgs_labels_by_file_pattern( filename , isTrueLabel , isTestData ):
    # example--> mhd_path_pattern = "K:/tianchi/train_*/*.mhd"


    pattern = "{}/*/*.jpg".format(filename)
    fileList = glob(pattern)

    x_datasets = np.ndarray([len(fileList), imgSize[0], imgSize[1], 3], dtype=np.uint8)
    if isTrueLabel:
        y_datasets = np.ones([len(fileList), ], dtype=np.uint8)
    else:
        y_datasets = np.zeros([len(fileList), ], dtype=np.uint8)
    for index, file in enumerate(fileList):
        im =  cv2.imread(file , cv2.IMREAD_COLOR)
        im =  cv2.cvtColor(im , cv2.COLOR_BGR2RGB)

        if im is not  None:
            im = misc.imresize(im, (imgSize[0], imgSize[1]))

        #Image.open(file)
        #Y = int(filename.replace(outputDataDir, ""))
        # img = cv.imread(file,1) #finally I did not use cv because cv seemed to have changed the channel data
        # print(img.shape)
        x_datasets[index] = im
        #y_datasets[i] = Y


    if isTestData == True:
        return  x_datasets , y_datasets
    else:
        num_images = len(fileList)
        #print  "num_images" , num_images
        rand_i = np.random.choice(range(num_images),size=num_images,replace=False)
        test_i = int(0.2*num_images)
        train_data = x_datasets[rand_i[test_i:]]
        train_label = y_datasets[rand_i[test_i:]]
        valid_data =  x_datasets[rand_i[0:test_i]]
        valid_label = y_datasets[rand_i[0:test_i]]
        return (train_data , train_label , valid_data ,valid_label)

def judge_condition(true_judge):
    return  true_judge > JUDGE_RATE


#def find_hard_simple(n)



esti_model_config = ModelConfig(EdLeNet, "dish_2_32_2_2048_0332",
                                [256, 192, 3], [5, 20, 2], [512 *4], 2,  # class 二分类
                                [1.0, 1.0])

if TrainingMode == True:
    #trainning

    # 测试400张的
    train_data_t, train_label_t, valid_data_t, valid_label_t = read_dataset_pipeline(
       DIR_TRUE_SAMPLE, isTrueLabel=True)

    print(train_data_t.shape, train_label_t.shape, valid_data_t.shape, valid_label_t.shape)

    train_data_f, train_label_f, valid_data_f, valid_label_f = read_dataset_pipeline(
        DIR_FALSE_SAMPLE, isTrueLabel=False)



    print(train_data_f.shape, train_label_f.shape, valid_data_f.shape, valid_label_f.shape)

    train_data = np.concatenate((train_data_t, train_data_f))
    train_label = np.concatenate((train_label_t, train_label_f))
    valid_data = np.concatenate((valid_data_t, valid_data_f))
    valid_label = np.concatenate((valid_label_t, valid_label_f))
    print(train_data.shape, train_label.shape, valid_data.shape, valid_label.shape)

    train_data = du.normalise_images(train_data, train_data)
    valid_data = du.normalise_images(valid_data, valid_data)

    me_c_sample_7x7 = ModelExecutor(esti_model_config, learning_rate=0.001)

    (tr_metrics, metrics, duration) = me_c_sample_7x7.train_model(train_data, train_label , valid_data,  valid_label, epochs=150
                                                            , batch_size=512)

#find hard sample
    #r = me_c_sample_7x7.test_model(test_data)



#test

else:

    #改成最好模型的名字
    esti_model_config.name = "dish_2_32_2_2048_0332_0.985652959783"



    # #测试正样本
    me_c_sample_7x7 = ModelExecutor(esti_model_config, learning_rate=0.001)
    #test_data_not, test_label_not = read_dataset_pipeline("./temp/test_data_1/" ,isTrueLabel=True , isTestData=True)
    #me_c_sample_7x7.test_model(test_data_not ,test_label_not , batch_size=512)

    to_be_trained_true = []
    fileList = glob("./temp/total_test_data_1/" + "*")
    true_judge_np = np.zeros(len(fileList) , dtype=np.float32)
    for index, file_name in enumerate(fileList):
        test_data , test_label  =  get_imgs_labels_by_file_pattern(file_name , isTrueLabel=True , isTestData=True  )
        test_data = du.normalise_images(test_data,test_data)
        r = me_c_sample_7x7.predict(test_data)
        print(file_name)
        binCount = np.bincount(r)
        true_judge = 0
        if len(binCount) == 2:
            true_judge = binCount[1] / 256.0
        print(true_judge)
        true_judge_np[index] = true_judge
        if judge_condition(true_judge) == False:
            to_be_trained_true.append(file_name)
    right_num = len(np.where(true_judge_np > JUDGE_RATE)[0])#这里0的意思是取出数组
    print(right_num , float(len(true_judge_np)))
    right_rate =  right_num / float(len(true_judge_np))
    print("统计识别率", right_rate)

    #测试负样本 0.7980, 也就是平均20
    #  test_data_not, test_label_not = read_dataset_pipeline("./temp/test_data_0/" ,isTrueLabel=False , isTestData=True)
    # me_c_sample_7x7.test_model(test_data_not ,test_label_not , batch_size=512)
    to_be_trained_false = []
    fileList = glob("./temp/total_test_data_0/" + "*")
    true_judge_np = np.zeros(len(fileList) , dtype=np.float32)

    for index, file_name in enumerate(fileList):
        test_data , test_label  =  get_imgs_labels_by_file_pattern(file_name , isTrueLabel=True , isTestData=True  )
        test_data = du.normalise_images(test_data,test_data)
        r = me_c_sample_7x7.predict(test_data)
        print(file_name)
        #我靠,需要考虑全部是0的情况
        binCount = np.bincount(r)
        true_judge = 0
        if len(binCount) == 2:
            true_judge = binCount[1] / 256.0
        print(true_judge)
        true_judge_np[index] = true_judge
        if judge_condition(true_judge):
            to_be_trained_false.append(file_name)

    false_right_num = len(np.where(true_judge_np > JUDGE_RATE)[0])#这里0的意思是取出数组
    print(false_right_num)
    #print "原来置信率", 37 / float(29 + 37)  #取yatedish文件夹统计

    print("统计识别率", right_rate)
    print("引入神经网络后置信率", right_num / float(false_right_num + right_num))

    print(to_be_trained_false , to_be_trained_true)