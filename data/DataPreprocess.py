#preprocess data
#author: Yunxuan
#date: 01/26/2016

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import csv

#import training data
with open('training_data.txt','r') as train_f:
    train_iter = csv.reader(train_f, delimiter='|')
    trainlist = [i for i in train_iter]
train_raw = np.asarray(trainlist)
Label = train_raw[0,:]
x_train = train_raw[1:,0:-1]
y_train = train_raw[1:,-1]
x_train = np.asarray(x_train, dtype='float64')
y_train = np.asarray(y_train, dtype='int')

#import testing data
with open('testing_data.txt','r') as test_f:
    test_iter = csv.reader(test_f, delimiter='|')
    testlist = [i for i in test_iter]
x_test = np.asarray(testlist[1:])
x_test = np.asarray(x_test, dtype='float64')

'''
from sklearn import preprocessing
x = np.zeros([x_train.shape[0]+x_test.shape[0], x_train.shape[1]])
x[0:x_train.shape[0],:] = x_train
x[x_train.shape[0]:,:] = x_test
x_norm = preprocessing.normalize(x, norm='l2')
x_norm_scale = preprocessing.scale(x_norm)
x_train = x_norm_scale[0:x_train.shape[0],:]
x_test = x_norm_scale[x_train.shape[0]:,:]
'''
