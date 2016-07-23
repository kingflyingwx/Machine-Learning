__author__ = 'wangxiang'
# -*- coding:utf8 -*-
import pickle
import os
import scipy.io as sio
from numpy import shape, concatenate, ones, asarray, dot


def loadDataSet(path):
    data = sio.loadmat(path)
    return data['X'], data['y']


def getAccurancy(dataX, dataY, res):
    print dataY
    resArr = asarray(res)
    predictRes = dot(dataX, resArr.T)  # 5000*10 , 每一行为一个样本预测的10个结果，取 最大的为预测结果 （训练样本维度000*401，类别为10）
    m, n = shape(predictRes)  # m为样本的个数
    predictClass = predictRes.argmax(axis=1) + 1
    resCount = predictClass == dataY.flat
    count = 0
    for i in resCount.flat:
        if i == True:
            count += 1
    return count * 1.0 / m


def predictOneVsAll(dataX, dataY):
    f = open("data/logistic_onevsall_model", "r")
    res_bin = pickle.load(f)
    res = pickle.loads(res_bin) 
    print "fmin_cg得到的模型,", res
    acc = getAccurancy(dataX, dataY, res)
    print "fmin_cg准确度为:", acc


def predictOneVsAll_fminbfgs(dataX, dataY):
    f = open("data/logistic_onevsall_model_fminbfgs", "r")
    res_bin = pickle.load(f)
    res = pickle.loads(res_bin)
    acc = getAccurancy(dataX, dataY, res)
    print "fmin_bfgs准确度为:", acc


if __name__ == "__main__":
    path = "data/logreg/handwritten_digit.mat"
    dataX, dataY = loadDataSet(path)
    print "数据集一部分:", dataX[0, 100:160]
    m, n = shape(dataX)
    dataX = concatenate((ones((m, 1)), dataX), axis=1)
    if os.path.isfile("data/logistic_onevsall_model") == True:
        predictOneVsAll(dataX, dataY)
    if os.path.isfile("data/logistic_onevsall_model_fminbfgs") == True:
        predictOneVsAll_fminbfgs(dataX, dataY)
    else:
        print "please train model"
