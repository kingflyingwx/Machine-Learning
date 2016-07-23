__author__ = 'wangxiang'
# -*- coding:utf8 -*-
from  matplotlib import pyplot as plt
from numpy import meshgrid, arange, shape, asarray


def showDataFigure(theta, dataMat, labelMat, lamda):
    X1, X2 = meshgrid(arange(-1.0, 1.5, 0.025), arange(-1.0, 1.5, 0.025))
    dataArr = asarray(dataMat)
    n = shape(dataArr)[0]
    print "记录条数：", n
    x1cord = [];
    y1cord = []
    x2cord = [];
    y2cord = []
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
    plt.xlabel('Exam1 Score')
    plt.ylabel('Exam2 Score')
    plt.contour(X1, X2, theta[0] + theta[1] * X1 + theta[2] * X2 + theta[3] * X1 ** 2 + theta[4] * X1 * X2 + theta[
        5] * X2 ** 2 + theta[6] * X1 ** 3 + theta[7] * (X1 ** 2) * X2 + theta[8] * X1 * (X2 ** 2) + theta[9] * (
                    X2 ** 3) + theta[10] * X1 ** 4 + theta[11] * (X1 ** 3) * X2 + theta[12] * (X1 ** 2) * (X2 ** 2) +
                theta[
                    13] * (X1 ** 1) * (X2 ** 3) + theta[14] * (X1 ** 0) * (X2 ** 4) + theta[15] * (X1 ** 5) * (
                    X2 ** 0) + theta[16] * (X1 ** 4) * (X2 ** 1) + theta[17] * (X1 ** 3) * (X2 ** 2) + theta[18] * (
                    X1 ** 2) * (X2 ** 3) + theta[19] * (X1 ** 1) * (X2 ** 4) + theta[20] * (X1 ** 0) * (X2 ** 5) +
                theta[
                    21] * (X1 ** 6) * (X2 ** 0) + theta[22] * (X1 ** 5) * (X2 ** 1) + theta[23] * (X1 ** 4) * (
                    X2 ** 2) + theta[24] * (X1 ** 3) * (X2 ** 3) + theta[25] * (X1 ** 2) * (X2 ** 4) + theta[26] * (
                    X1 ** 1) * (X2 ** 5) + theta[27] * (X1 ** 0) * (X2 ** 6), [0])
    plt.title("lamda=1")
    plt.show()