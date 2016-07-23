__author__ = 'wangxiang'
# -*- coding:utf8 -*-
from numpy import array, shape, ones
import sys
from svmutil import *
import scipy.io as sio
import matplotlib.pyplot as plt


class SVM:
    X = array([])
    y = []

    def __init__(self, path, pathTest, d):
        data = sio.loadmat(path)
        self.X = data['X']
        self.y = data['y']
        data = sio.loadmat(pathTest)
        self.testX = data['Xtest']
        self.testY = data['ytest']
        print "test训练集合:", data
        print type(self.testX), shape(self.testX)
        print self.testX[0, :], type(self.testX[0, :]), shape(self.testX[0, :])
        self.testNewX = d
        self.testNewY = ones((1, 1))
        print "testNewY", self.testNewY

    def processData(self):
        f = open("data/svm/libSvmData", "w")
        for i in range(shape(self.X)[0]):
            res = str(self.y[i, 0]) + " "
            for j in range(shape(self.X)[1]):
                res += str(j + 1) + ":" + str(self.X[i, j]) + " "
            f.write(res)
            f.write("\n")
            # f.write("%s %s %s" % self.X[0,0])
        f.close()
        f = open("data/svm/libSvmTestData", "w")
        for i in range(shape(self.testX)[0]):
            res = str(self.testY[i, 0]) + " "
            for j in range(shape(self.testX)[1]):
                res += str(j + 1) + ":" + str(self.testX[i, j]) + " "
            f.write(res)
            f.write("\n")
            # f.write("%s %s %s" % self.X[0,0])
        f.close()
        f = open("data/svm/libSvmTestNewData", "w")
        for i in range(shape(self.testNewX)[0]):
            res = str(self.testNewY[i, 0]) + " "
            for j in range(shape(self.testNewX)[1]):
                res += str(j + 1) + ":" + str(self.testNewX[i, j]) + " "
            f.write(res)
            f.write("\n")
            # f.write("%s %s %s" % self.X[0,0])
        f.close()
        # f=open("data/libSvmData2","w")
        # for i in range(shape(self.X)[0]):
        # if self.y[i,0]==0:
        # res="-1 "
        # #    else:
        # res="+1 "
        #   for j in range(shape(self.X)[1]):
        #       res+=str(j+1)+":"+str(self.X[i,j])+" "

        #   f.write(res)
        #   f.write("\n")
        #f.write("%s %s %s" % self.X[0,0])
        #f.close()

    def trainModel(self, c, t):
        y, x = svm_read_problem("data/svm/libSvmData")
        m = svm_train(y, x, "-c " + str(c) + " -t " + str(t))
        f = open("data/svm/svnmodel", "w")
        f.write(str(m))
        f.close()
        p_label, p_acc, p_val = svm_predict(y, x, m)
        # y,x=svm_read_problem("data/libSvmData2")
        # m=svm_train(y,x,"-c 1 -t 0")
        # p_label, p_acc, p_val =svm_predict(y,x,m)
        print "@@@@@@@train集p_label@@@@@@@", p_label
        print "@@@@@@@train集p_acc@@@@@@", p_acc
        print "@@@@@@train集p_val@@@@@@@@", p_val
        y, x = svm_read_problem("data/libSvmTestData")
        p_label, p_acc, p_val = svm_predict(y, x, m)
        # y,x=svm_read_problem("data/libSvmData2")
        # m=svm_train(y,x,"-c 1 -t 0")
        # p_label, p_acc, p_val =svm_predict(y,x,m)
        print "@@@@@@@test集p_label@@@@@@@", p_label
        print "@@@@@@@test集p_acc@@@@@@", p_acc
        print "@@@@@@test集p_val@@@@@@@@", p_val
        y, x = svm_read_problem("data/libSvmTestNewData")
        p_label, p_acc, p_val = svm_predict(y, x, m)
        #y,x=svm_read_problem("data/libSvmData2")
        #m=svm_train(y,x,"-c 1 -t 0")
        #p_label, p_acc, p_val =svm_predict(y,x,m)
        print "@@@@@@@testNew p_label@@@@@@@", p_label
        print "@@@@@@@testNew p_acc@@@@@@", p_acc
        print "@@@@@@testNew p_val@@@@@@@@", p_val

        #y,x=svm_read_problem("data/libSvmData2")
        #m=svm_train(y,x,"-c 1 -t 0")
        #p_label, p_acc, p_val =svm_predict(y,x,m)
        #print "@@@@@@@@@@@@@@@@@@@@@@@",p_label
        #print "@@@@@@@@@@@@@@@@@@@@",p_acc
        #print "@@@@@@@@@@@@@@@@@@@@",p_val

    def showDataFigure(self):
        m = shape(self.X)[0]
        x1cord = [];
        x2cord = [];
        y1cord = [];
        y2cord = []
        for i in range(m):
            if self.y[i] == 1:
                x1cord.append(self.X[i, 0])
                y1cord.append(self.X[i, 1])
            else:
                x2cord.append(self.X[i, 0])
                y2cord.append(self.X[i, 1])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        admitted = ax.scatter(x1cord, y1cord, s=30, c='black', marker='+')
        nadmit = ax.scatter(x2cord, y2cord, s=30, c='yellow', marker='o')
        # plt.legend((admitted,nadmit),('admitter','not admitted'), scatterpoints=1)  # 加图例
        # ax.scatter(xcord, ycord, s=30, c='red', marker='s')
        plt.show()


if __name__ == "__main__":
    # y,x=svm_read_problem('D:/wx_algorithm/libsvm-3.20/heart_scale')
    # print y
    # print x
    # m=svm_train(y[:200],x[:200],'-c 4')
    # print m
    #p_label,p_acc,p_val=svm_predict(y[200:],x[200:] ,m)
    #print p_label
    #print p_acc
    #print p_val
    obj1 = SVM("data/svm/ex6data1.mat")
    #print obj1.X
    #obj1.showDataFigure()
    obj1.processData()
    c = 1
    t = 0
    obj1.trainModel(c, t)
    c = 100
    obj1.trainModel(c, t)
    obj2 = SVM("data/svm/ex6data2.mat")
    obj2.processData()
    c = 1
    t = 2
    obj2.trainModel(c, t)
    c = 10000
    obj2.trainModel(c, t)