__author__ = 'wangxiang'
# -*- encoding:utf8 -*-
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
  加载训练数据
'''


def loadDataSet():
    dataMat = []
    labelMat = []
    data = open('data/linereg/liner_onevar.data')
    for line in data:
        lineArr = line.strip().split(",")
        # dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        dataMat.append([1.0, float(lineArr[0])])
        labelMat.append([float(lineArr[1])])
    return dataMat, labelMat


'''
图形显示
'''

def showDataFigure(dataMat, labelMat, times, theta):
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    print "记录条数：", n
    xcord = [];
    ycord = []
    for i in range(n):
        xcord.append(dataArr[i, 1])
        ycord.append(labelMat[i])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord, ycord, s=30, c='red', marker='s')
    if times == 2:
        x = arange(5, 24, 1)
        y = theta[0, 0] + theta[1, 0] * x
        ax.plot(x, y)
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()


def showDataFigure2(dataMat, labelMat, times, theta):
    dataMat = array(dataMat)
    n = shape(dataMat)[0]
    print "记录条数：", n
    x1cord = []
    x2cord = []
    ycord = []
    for i in range(n):
        x1cord.append(dataMat[i, 1])
        x2cord.append(dataMat[i, 2])
        ycord.append(labelMat[i])
    fig = plt.figure()
    ax = Axes3D(fig)
    # ax = fig.add_subplot(111)
    ax.scatter(x1cord, x2cord, ycord, s=30, c='red', marker='s')  # scatter 画点
    if times == 2:
        x1 = arange(-0.5, 0.5, 0.01)
        x2 = arange(-0.5, 0.5, 0.01)
        [x1, x2] = meshgrid(x1, x2)
        y = theta[0, 0] + theta[1, 0] * x1 + theta[2, 0] * x2

        ax.plot_surface(x1, x2, y, rstride=8, cstride=8, alpha=0.1)
        # ax.plot(x1,x2,y)

    # plt.xlabel('Population of City in 10,000s')
    # plt.ylabel('Profit in $10,000s')
    plt.show()


def showDataFigure_NormalEquation(dataMat, labelMat, times, theta):
    dataMat = array(dataMat)
    n = shape(dataMat)[0]
    print "记录条数：", n
    x1cord = []
    x2cord = []
    ycord = []
    for i in range(n):
        x1cord.append(dataMat[i, 1])
        x2cord.append(dataMat[i, 2])
        ycord.append(labelMat[i])
    fig = plt.figure()
    ax = Axes3D(fig)
    # ax = fig.add_subplot(111)
    ax.scatter(x1cord, x2cord, ycord, s=30, c='red', marker='s')  # scatter 画点
    if times == 2:
        x1 = arange(0, 4000, 10)
        x2 = arange(0, 9, 1)
        [x1, x2] = meshgrid(x1, x2)
        y = theta[0, 0] + theta[1, 0] * x1 + theta[2, 0] * x2

        ax.plot_surface(x1, x2, y, rstride=8, cstride=8, alpha=0.1)
        # ax.plot(x1,x2,y)

    # plt.xlabel('Population of City in 10,000s')
    # plt.ylabel('Profit in $10,000s')
    plt.show()


def computeCostFunction(dataMat, labelMat, theta):
    dataArr = array(dataMat)
    # print shape(dataArr),shape(theta),shape(labelMat),shape(dataMat)
    labelArr = array(labelMat)
    # print shape(labelArr)
    # print "labelArr.transpose()",labelArr.transpose()
    # print labelArr
    n = shape(labelArr)[0]
    #print n
    # print "dataArr:\t",dataArr,"\ntheta:\t",theta
    hx = dot(dataArr, theta)
    #print shape(hx)
    #print "hx",hx
    err = hx - labelArr  #hx-y
    #print err
    Jtheta = dot(err.transpose(), err) / (2 * n)
    return Jtheta


'''
Batch Gradient Descent
'''


def gradientDescent(dataMat, labelMat, theta, iterations, alpha):
    Jt = []
    dataArr = array(dataMat)
    labelArr = array(labelMat)
    # print dataArr
    # print labelArr
    n = shape(labelArr)[0]
    dataArrTran = dataArr.transpose()
    hx = dot(dataArr, theta)
    err = hx - labelArr
    sumpart = dot(dataArrTran, err)
    for i in range(iterations):
        theta = theta - alpha * sumpart / n
        Jtheta = computeCostFunction(dataMat, labelMat, theta)
        Jt.append([Jtheta[0, 0]])
        # Jtheta.append(computeCostFunction(dataMat,labelMat,theta))
        # print "第" + str(i) + "次循环Jtheta："
        #print Jtheta
        hx = dot(dataArr, theta)
        err = hx - labelArr
        sumpart = dot(dataArrTran, err)
    # print type(Jtheta),shape(Jtheta)
    # print type(Jtheta[0,0])
    #print Jt
    #print array(Jt),shape(array(Jt))
    return theta, Jt


def figJthetaPerInter(Jt):
    iterNum = range(1, 1501)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(iterNum, Jt, s=10, c='red', marker='s')
    plt.xlabel('iteration nums')
    plt.ylabel('costFunction J(theta)')
    plt.show()


def loadDataSet2():
    dataMat = [];
    labelMat = []
    data = open('data/linereg/liner_multivar.data')
    for line in data:
        lineArr = line.strip().split(",")
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append([float(lineArr[2])])
    return dataMat, labelMat


def featureScaling(dataMat):
    dataArr = array(dataMat)
    m = shape(dataArr)[0]  # m代表记录条数
    n = shape(dataArr)[1]  # n 代表特征数目
    avgFea = []
    maxFea = []
    minFea = []
    dataArr2 = dataArr[:, 1:n]
    for i in range(n - 1):
        avgFea.append(sum(dataArr2[:, i]) * 1.0 / m)
        maxFea.append(max(dataArr2[:, i]))
        minFea.append(min(dataArr2[:, i]))
    # print avgFea,maxFea,minFea,shape(array(avgFea))
    # print array(avgFea),array(maxFea)
    feaAvgArr = array(avgFea)
    maxFeaArr = array(maxFea)
    minFeaArr = array(minFea)
    print feaAvgArr
    print maxFeaArr
    print minFeaArr
    # print shape(dataArr2)
    dataFeaScalMat = (dataArr2 - feaAvgArr) / (maxFeaArr - minFeaArr)
    #print dataFeaScalMat
    onesArr = ones((m, 1))
    dataArrFinal = concatenate((onesArr, dataFeaScalMat), axis=1)
    return dataArrFinal, feaAvgArr, maxFeaArr, minFeaArr


def NormalEquation(dataMat, labelMat, theta):
    dataArr = array(dataMat)
    labelArr = array(labelMat)
    xtx = dot(dataArr.transpose(), dataArr)
    xtxtran = mat(xtx).I
    xtxtranxtran = dot(xtxtran, dataArr.transpose())
    theta = dot(xtxtranxtran, labelArr)
    print "normal equation 得到的theta结果为：", theta
    return theta


def predictFunction(sample, theta, feaAvgArr, maxFeaArr, minFeaArr):
    sampleArr = array(sample)
    # print feaAvgArr, maxFeaArr, minFeaArr
    n = len(sampleArr)
    sampleArrFeaScal = (sampleArr[1:n] - feaAvgArr) / (maxFeaArr - minFeaArr)
    # print sampleArrFeaScal
    onesArr = ones((1))
    #print onesArr,type(onesArr),shape(onesArr)
    onesArr2 = ones((1, 1))
    #print onesArr2,type(onesArr2),shape(onesArr2)
    sampleArrFinal = concatenate((onesArr, sampleArrFeaScal), axis=1)
    #print sampleArrFinal
    #print theta,theta[2,0]
    result = sampleArrFinal[0] * theta[0, 0] + theta[1, 0] * sampleArrFinal[1] + theta[2, 0] * sampleArrFinal[2]
    print "通过梯度下降方法获得的预测值为:", result


def predictFunction_NormalEquation(sample, theta):
    result = sample[0] * theta[0, 0] + theta[1, 0] * sample[1] + theta[2, 0] * sample[2]
    print "通过normal equation方法获得的预测值为", result


if __name__ == "__main__":
    dataMat, labelMat = loadDataSet()
    print "训练数据为:", dataMat, '\n', labelMat
    theta = zeros((2, 1))
    # print type(theta)
    showDataFigure(dataMat, labelMat, 1, theta)
    iterations = 1500
    alpha = 0.01
    Jtheta = computeCostFunction(dataMat, labelMat, theta)
    print "初始Jtheta："
    print Jtheta
    theta, Jt = gradientDescent(dataMat, labelMat, theta, iterations, alpha)
    figJthetaPerInter(Jt)
    print "最终训练得到的theta向量为："
    print theta, type(theta)
    showDataFigure(dataMat, labelMat, 2, theta)

    iterations = 1500
    alpha = 0.01
    # ####################################################################
    # ###############################多变量###############################
    dataMat, labelMat = loadDataSet2()
    theta = zeros((3, 1))
    showDataFigure2(dataMat, labelMat, 1, theta)
    dataMatFinal, feaAvgArr, maxFeaArr, minFeaArr = featureScaling(
        dataMat)  #feaAvgArr,maxFeaArr,minFeaArr 对应各个特征值的平局值，最大值以及最小值
    Jtheta = computeCostFunction(dataMatFinal, labelMat, theta)
    print "初始Jtheta:"
    print Jtheta
    theta, Jt = gradientDescent(dataMatFinal, labelMat, theta, iterations, alpha)
    print "最终训练得到的theta向量为："
    print theta, type(theta)
    print "通过梯度下降方法最终的jtheta值为:"
    print Jt[-1]
    #figJthetaPerInter(Jt)
    showDataFigure2(dataMatFinal, labelMat, 2, theta)
    sample = [1.0, 1650.0, 3.0]
    predictFunction(sample, theta, feaAvgArr, maxFeaArr, minFeaArr)
    theta = zeros((3, 1))
    theta = NormalEquation(dataMat, labelMat, theta)
    print "通过normal equation最终的jtheta值为:"  # 通过比较发现通过normal equation比梯度下降更好的拟合了数据。所以当特征值不是很多的时候，例如小于10000， 线性回归完全可以用normal eqution的方法来做
    Jtheta = computeCostFunction(dataMat, labelMat, theta)
    print Jtheta
    showDataFigure_NormalEquation(dataMat, labelMat, 2, theta)
    predictFunction_NormalEquation(sample, theta)