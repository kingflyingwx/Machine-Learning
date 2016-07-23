__author__ = 'wangxiang'
# -*- coding:utf8 -*-
import matplotlib.pyplot as plt
import LogisticPic
from numpy import *
from scipy.optimize import fmin_bfgs, minimize


def loadDataSet(path):
    dataMat = []
    labelMat = []
    f = open(path)
    for line in f.readlines():
        data = line.split(",")
        dataMat.append([1.0, float(data[0]), float(data[1])])
        labelMat.append([int(data[2])])
    # print dataMat
    # print labelMat
    return dataMat, labelMat


def showDataFigure(dataMat, labelMat, *args):
    dataArr = array(dataMat)
    # print dataArr
    n = shape(dataArr)[0]
    print "记录条数：", n
    x1cord = [];
    y1cord = []
    x2cord = [];
    y2cord = []
    # print type(labelMat[1])
    for i in range(n):
        if labelMat[i] == [1]:
            x1cord.append(dataArr[i, 1])
            y1cord.append(dataArr[i, 2])
        else:
            x2cord.append(dataArr[i, 1])
            y2cord.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    admitted = ax.scatter(x1cord, y1cord, s=30, c='black', marker='+')
    nadmit = ax.scatter(x2cord, y2cord, s=30, c='yellow', marker='o')
    plt.legend((admitted, nadmit), ('admitter', 'not admitted'), scatterpoints=1)  # 加图例
    if args:
        x = arange(30, 100, 5)
        y = -args[0] * 1.0 / args[2] - args[1] * x * 1.0 / args[2]
        ax.plot(x, y)
    plt.xlabel('Exam1 Score')
    plt.ylabel('Exam2 Score')
    plt.show()


def sigmodFunction(dataMat, theta):
    dataArr = array(dataMat)
    z = dot(dataArr, theta)
    p = 1e-5
    hx = 1 / (1 + exp(-z))
    hx_new = []
    for f in hx.flat:
        if f == 1:
            hx_new.append([f - p])
        elif f == 0:
            hx_new.append([f + p])
        else:
            hx_new.append([f])
    hx = asarray(hx_new)
    return hx


def getGradient(dataArr, labelArr, theta, m):
    X = dataArr.transpose()
    hx = sigmodFunction(dataArr, theta)
    hx_minux_y = hx - labelArr
    gra = dot(X, hx_minux_y) * 1.0 / m  # dot(array(dataMat).transpose(),hx-array(labenMat))*1.0/m
    return gra


def getGradient2(theta, *args):
    # print "getGradient2 中theta为：",theta
    dataArr = array(args[0])
    labelArr = array(args[1])
    m = shape(dataArr)[0]
    X = dataArr.transpose()
    # print shape(X)
    #print shape(theta),theta,type(theta),theta[0],theta[1],theta[2],len(theta)
    theta_new = []
    for i in theta:
        theta_new.append([i])
    theta = asarray(theta_new)
    #print "getGradient2 中theta经过转换为：",theta
    hx = sigmodFunction(dataArr, theta)
    hx_minux_y = hx - labelArr
    gra = dot(X, hx_minux_y) * 1.0 / m  #dot(array(dataMat).transpose(),hx-array(labenMat))*1.0/m
    return gra.flatten()


def getGradient_Regular(theta, *args):
    # print "getGradient2 中theta为：",theta
    dataArr = asarray(args[0])
    labelArr = asarray(args[1])
    lamda = args[2]
    m = shape(dataArr)[0]
    X = dataArr.transpose()
    theta_new = []
    for i in theta:
        theta_new.append([i])
    theta = asarray(theta_new)
    hx = sigmodFunction(dataArr, theta)
    hx_minux_y = hx - labelArr
    gra = dot(X, hx_minux_y) * 1.0 / m  # dot(array(dataMat).transpose(),hx-array(labenMat))*1.0/m
    gra = gra + lamda * 1.0 * theta / m
    gra[0, 0] = dot(X[0, :], hx_minux_y) * 1.0 / m
    return gra.flatten()


def costFunction(dataMat, labelMat, theta):
    dataArr = array(dataMat)
    labelArr = array(labelMat)
    m = shape(dataArr)[0]
    hx = sigmodFunction(dataArr, theta)
    loghx = ma.log(hx)
    # print "log hx:",loghx
    yhx = dot(loghx.transpose(), labelArr)
    # print "yhx:",yhx
    log1_hx = ma.log(1 - hx)
    #print "log1_hx:",log1_hx
    #print (-yhx-dot(log1_hx.transpose(),(1-labelArr)))*1.0/m
    jtheta = (-yhx - dot(log1_hx.transpose(), (1 - labelArr))) * 1.0 / m
    gra = getGradient(dataArr, labelArr, theta, m)
    print type(jtheta), type(gra.flatten())
    #print gra
    return jtheta, gra


def costFunction2(theta, *args):
    print "现在在调用cost函数"
    # dataArr=array(dataMat)
    dataArr = array(args[0])
    # print "costFunction2中dataArr值为:",dataArr
    #labelArr=array(labelMat)
    labelArr = array(args[1])
    #print "costFunction2中labelArr值为:",labelArr
    m = shape(dataArr)[0]
    #print "记录条数:",m
    theta_new = []
    for i in theta:
        theta_new.append([i])
    theta = array(theta_new)
    print "costFunction2中theta的值为：", theta
    hx = sigmodFunction(dataArr, theta)
    #print "初始hx值",hx
    loghx = ma.log(hx)
    #print "log hx:",loghx
    yhx = dot(loghx.transpose(), labelArr)
    #print "yhx:",yhx
    log1_hx = ma.log(1 - hx)
    #print "log1_hx:",log1_hx
    #print (-yhx-dot(log1_hx.transpose(),(1-labelArr)))*1.0/m
    jtheta = (-yhx - dot(log1_hx.transpose(), (1 - labelArr))) * 1.0 / m
    #gra=getGradient(dataArr,labelArr,theta,m)
    #print type(jtheta),type(gra.flatten())
    #print gra
    #print "###################",type(array(jtheta)[0]),array(jtheta)[0]
    print "###########################!!!!!!!!!!!!!!!!costFunction2得到的jtheta值为:", type(jtheta.flatten()[0]), jtheta, \
        jtheta.flatten()[0]
    return jtheta.flatten()[0]


# def decorated_cost(theta):
# return costFunction(dataMat, labelMat,theta)
def costFunction_Regular(theta, *args):
    print "现在在调用正则化的cost函数"
    dataArr = asarray(args[0])  #args0 为特征数据  args1 为类别数据
    labelArr = array(args[1])
    m = shape(dataArr)[0]
    #print "记录条数:",m
    theta_new = []
    for i in theta:  # 这段可以用reshape方法搞定
        theta_new.append([i])
    theta = asarray(theta_new)
    print "costFunction_Regular中theta的值为：", theta
    hx = sigmodFunction(dataArr, theta)  #计算预测的类别概率值
    loghx = ma.log(hx)
    #print "log hx:",loghx
    yhx = dot(loghx.transpose(), labelArr)
    #print "yhx:",yhx
    log1_hx = ma.log(1 - hx)
    #print "log1_hx:",log1_hx
    #print (-yhx-dot(log1_hx.transpose(),(1-labelArr)))*1.0/m
    print "lameda 值为:", args[2]  # args[2]为传入的lameda参数
    jtheta = (-yhx - dot(log1_hx.transpose(), (1 - labelArr))) * 1.0 / m + args[2] * 1.0 / (2 * m) * (
        dot(theta.transpose(), theta) - theta[0, 0] ** 2)
    #gra=getGradient(dataArr,labelArr,theta,m)
    #print type(jtheta),type(gra.flatten())
    #print gra
    #print "###################",type(array(jtheta)[0]),array(jtheta)[0]
    print "costFunction2得到的jtheta值为:", type(jtheta.flatten()), type(jtheta.flatten()[0]), jtheta, type(jtheta), shape(
        jtheta), jtheta.flatten()[0]
    return jtheta.flatten()[0]


def predictRes(x, theta, y, threshold):
    dataArr = asarray(x)
    labelArr = asarray(y)
    n = shape(y)[0]
    #print theta
    z = dot(x, theta)
    p = 1.0 / (1 + exp(-1 * z))
    p[p < threshold] = 0
    p[p >= threshold] = 1
    #print p
    res = p - labelArr
    count = 0
    for r in res.flat:
        if r != 0:
            count += 1
    print count
    print n
    return (n - count * 1.0) / n


def mapFeature(dataMat, degree):
    dataArr = asarray(dataMat)
    dataNew = ones(size(dataArr[:, 0]))
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            dataNew = concatenate((dataNew, (dataArr[:, 1] ** (i - j)) * (dataArr[:, 2] ** j)))
    dataNew = reshape(dataNew, (118, 28), order='F')
    print shape(dataNew)
    return dataNew


if __name__ == "__main__":
    print "begin logistic regression"
    path = "data/logreg/logistic_data1.data"
    dataMat, labelMat = loadDataSet(path)
    #print "原始数据行数为:",len(dataMat)
    #print shape(dataMat)
    showDataFigure(dataMat, labelMat)
    theta = zeros((3, 1))
    #print "要开始调用  fmin_bfgs:",theta
    xopt = fmin_bfgs(costFunction2, theta, fprime=getGradient2, args=(dataMat, labelMat), maxiter=400, full_output=1,
                     retall=1)
    print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^", xopt
    showDataFigure(dataMat, labelMat, *xopt[0])
    #xArr=asarray(x)
    #predictRes(xArr,xopt[0])
    #print xopt[0].transpose()
    n = len(dataMat[0])
    theta = xopt[0].reshape((n, 1))
    threshold = 0.5
    accurancy = predictRes(dataMat, theta, labelMat, threshold)
    print "准确度为:", accurancy
    #########################################################################################################################
    ##################################带有regularized的logistic regression 算法#########################
    path = "data/logreg/logistic_data2.data"
    dataMat, labelMat = loadDataSet(path)
    showDataFigure(dataMat, labelMat)
    degree = 6
    dataNew = mapFeature(dataMat, degree)
    print "^^^^^^^^^^^^^dataNew^", dataNew

    theta = zeros((28, 1)).flatten()
    #jtheta=costFunction_Regular(theta,*(dataNew,labelMat,1))
    #print jtheta
    lamda = 1
    xopt = fmin_bfgs(costFunction_Regular, theta, fprime=getGradient_Regular, args=(dataNew, labelMat, lamda),
                     maxiter=400, full_output=1, retall=0)
    print xopt[0], xopt[1]
    LogisticPic.showDataFigure(xopt[0], dataMat, labelMat, lamda)

