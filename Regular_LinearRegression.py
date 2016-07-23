__author__ = 'wangxiang'
# -*- coding:utf8 -*-
import scipy.io as sio
from numpy import shape, asarray, empty, array, concatenate, ones, dot, reshape, arange, zeros, meshgrid, std
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cg


class RegularizedLinearRegression:
    X = array([])
    y = []
    Xval = []
    yval = []
    Xtest = []
    ytest = []
    theta = []
    Xm = 0
    Xn = 0
    lamda = 1
    res = []
    Xval_m = 0
    Xval_n = 0
    Xtest_m = 0
    Xtest_n = 0

    def __init__(self, path):
        self.theta = ones((2, 1))
        self.lamda = 0
        data = sio.loadmat(path)
        self.X = data["X"]
        self.y = data['y']
        self.Xm, self.Xn = shape(self.X)
        self.Xtest = data["Xtest"]
        self.ytest = data["ytest"]
        self.Xval = data["Xval"]
        self.yval = data["yval"]
        self.Xval_m, self.Xval_n = shape(self.Xval)
        self.Xtest_m, self.Xtest_n = shape(self.Xtest)
        # print X,type(X),shape(X)
        #print y,Xtest,yval

    def addBiasToData(self):  # 增加bias
        self.X = concatenate((ones((self.Xm, 1)), self.X), axis=1)
        self.Xn = shape(self.X)[1]
        self.Xval = concatenate((ones((self.Xval_m, 1)), self.Xval), axis=1)
        self.Xval_n = shape(self.Xval)[1]
        self.Xtest = concatenate((ones((self.Xtest_m, 1)), self.Xtest), axis=1)
        self.Xtest_n = shape(self.Xtest)[1]

    def computeCostFunction(self, theta, *args):
        # print "%%%%%%%%theta%%%%%%%%%",theta
        X = args[0]
        y = args[1]
        lamda = args[2]
        Xm, Xn = shape(X)
        theta = reshape(theta, (Xn, 1))
        hx = dot(X, theta)
        err = hx - y  #hx-y
        theta_square = dot(theta[1, :].T, theta[1, :])  # bias 不用正则化
        Jtheta = dot(err.transpose(), err) / (2 * Xm) + lamda * 1.0 * theta_square / (2 * Xm)
        return Jtheta.flatten()[0]

    def showDataFigure(self, X, p, feaAvgArr, maxFeaArr, minFeaArr):
        xcord = []
        ycord = []
        for i in range(self.Xm):
            xcord.append(self.X[i, 1])
            ycord.append(self.y[i, 0])
        # print "xcord",xcord
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xcord, ycord, s=30, c='red', marker='s')

        x = arange(-80L, 60L, 1)
        if shape(X)[1] > 2:  # 多项式回归时，要对测试集按照训练集的 feaAvgArr, maxFeaArr, minFeaArr 进行归一化
            print "process   @@@@@@@"
            x_feasca = reshape(x, (shape(x)[0], 1))
            midX = x_feasca
            for i in range(2, p + 1):
                x_feasca = concatenate((x_feasca, midX ** i), axis=1)
            print "feaAvgArr:", feaAvgArr
            x_feasca = (x_feasca - feaAvgArr) / (maxFeaArr - minFeaArr)
            x_feasca = concatenate((ones((shape(x_feasca)[0], 1)), x_feasca), axis=1)
            #print "##############",self.res[0]
            #print x
            res = reshape(self.res[0], (1, shape(self.res[0])[0]))
            print res
            y = dot(self.res[0], x_feasca.T)
            #for i in range(shape(X)[1]):
            #    y=self.res[0][i]*(x**i)
            #    print "yyyyyy",y
            #y=self.res[0][0]+self.res[0][1]*(x)+self.res[0][2]*(x**2)+self.res[0][3]*(x**3)+self.res[0][4]*(x**4)+self.res[0][5]*(x**5)+self.res[0][6]*(x**6)+self.res[0][7]*(x**7)+self.res[0][8]*(x**8)
            #print y
        else:
            y = self.res[0][0] + self.res[0][1] * x
        ax.plot(x, y)
        plt.axis([-80, 60, -30, 40])
        plt.xlabel('Change in water level (x)')
        plt.ylabel('Water flowing out of the dam(y)')
        plt.show()

    def gradientDescent(self, theta, *args):
        X = args[0]
        y = args[1]
        lamda = args[2]
        Xm, Xn = shape(X)
        theta = reshape(theta, (Xn, 1))
        hx = dot(X, theta)
        err = hx - y
        sumpart = dot(X.T, err)
        partDeri = sumpart * 1.0 / Xm + lamda * theta * 1.0 / Xm
        partDeri[0, :] = partDeri[0, :] - lamda * theta[0, :] * 1.0 / Xm
        return partDeri.flatten()

    def fminCg(self, theta, X, y, lamda):
        self.res = fmin_cg(self.computeCostFunction, theta, self.gradientDescent, (X, y, lamda), maxiter=400,
                           full_output=1, retall=1)

    # def trainLinearReg(self):

    def learnCurves(self):
        Jtheta_TrainSets = []
        Jtheta_CvSets = []
        for i in range(1, self.Xm + 1):
            self.fminCg(self.theta, self.X[0:i, :], self.y[0:i, :], self.lamda)
            #计算训练误差（set lamda=0 and 使用部分训练集）和交叉验证集误差（使用全部的交叉验证集）
            Jtheta_TrainSets.append(self.computeCostFunction(self.res[0], *(self.X[0:i, :], self.y[0:i, :], 0)))
            Jtheta_CvSets.append(self.computeCostFunction(self.res[0], *(self.Xval, self.yval, 0)))
        return Jtheta_TrainSets, Jtheta_CvSets

    def learnCurvesPoly(self, theta, X, y, Xval, yval):
        Jtheta_TrainSets = []
        Jtheta_CvSets = []
        for i in range(1, self.Xm + 1):
            XFeaScaFinal, feaAvgArr, maxFeaArr, minFeaArr = self.featureScaling(X[0:i, :])
            #print "内层的X",X[0:i,:]
            #print "内层的XFeaScaFinal",XFeaScaFinal
            self.fminCg(theta, XFeaScaFinal, y[0:i, :], self.lamda)
            #计算训练误差（set lamda=0 and 使用部分训练集）和交叉验证集误差（使用全部的交叉验证集）
            Jtheta_TrainSets.append(self.computeCostFunction(self.res[0], *(XFeaScaFinal, y[0:i, :], 0)))
            xval_feasca = self.polyScaling(Xval, feaAvgArr, maxFeaArr, minFeaArr)
            Jtheta_CvSets.append(self.computeCostFunction(self.res[0], *(xval_feasca, yval, 0)))
        return Jtheta_TrainSets, Jtheta_CvSets

    def showLearnCurves(self, Jtheta_TrainSets, Jtheta_CvSets):
        iterNum = range(1, len(Jtheta_TrainSets) + 1)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(iterNum, Jtheta_TrainSets, c='red', marker='s')
        ax.plot(iterNum, Jtheta_CvSets, c='green', marker='s')
        plt.xlabel('iteration nums')
        plt.ylabel('costFunction J(theta)')
        plt.show()

    def polyFeatures(self, p, X):
        X_copy = X
        for i in range(2, p + 1):
            X_copy = concatenate((X_copy, X[:, 1:] ** i), axis=1)
        return X_copy

    def polyScaling(self, X, feaAvgArr, maxFeaArr, minFeaArr):
        xval_feasca = (X[:, 1:] - feaAvgArr) / (maxFeaArr - minFeaArr)
        xval_feasca = concatenate((ones((shape(xval_feasca)[0], 1)), xval_feasca), axis=1)
        return xval_feasca

    def featureScaling(self, X):
        m, n = shape(X)  # m代表记录条数    n 代表特征数目
        avgFea = []
        maxFea = []
        minFea = []
        #stdFeaArr=[]
        X = X[:, 1:n]
        #print "featureScaling:" ,X
        for i in range(n - 1):
            avgFea.append(sum(X[:, i]) * 1.0 / m)
            maxFea.append(max(X[:, i]))
            minFea.append(min(X[:, i]))
        #print "feature scaling avg:", avgFea
        feaAvgArr = asarray(avgFea)
        maxFeaArr = asarray(maxFea)
        minFeaArr = asarray(minFea)
        #stdFeaArr=std(feaAvgArr)
        if m != 1:

            XFeaScal = (X - feaAvgArr) / (maxFeaArr - minFeaArr)
        else:
            XFeaScal = (X - feaAvgArr)
        #XFeaScal = (X - feaAvgArr) / stdFeaArr
        onesArr = ones((m, 1))
        XFeaScaFinal = concatenate((onesArr, XFeaScal), axis=1)
        #return dataArrFinal, feaAvgArr, maxFeaArr, minFeaArr
        return XFeaScaFinal, feaAvgArr, maxFeaArr, minFeaArr

    def setLamda(self, lamda):
        self.lamda = lamda


if __name__ == "__main__":
    obj1 = RegularizedLinearRegression("data/linereg/ex5data1.mat")
    obj1.addBiasToData()
    # print "@@@@@@@@@@@@@@@@X",obj1.X
    #obj1.showDataFigure(obj1.X,obj1.y)
    #jtheta=obj1.computeCostFunction(obj1.theta)
    #graDes=obj1.gradientDescent(obj1.theta)
    obj1.fminCg(obj1.theta, obj1.X, obj1.y, obj1.lamda)
    obj1.showDataFigure(obj1.X, 1, 1, 1, 1)
    Jtheta_TrainSets, Jtheta_CvSets = obj1.learnCurves()
    ####画出learningcurves
    obj1.showLearnCurves(Jtheta_TrainSets, Jtheta_CvSets)
    p = 8
    X = obj1.polyFeatures(p, obj1.X)  # 简单的 1+x+x**2+x**3+x**4+......+x**p
    XFeaScaFinal, feaAvgArr, maxFeaArr, minFeaArr = obj1.featureScaling(X)
    theta = ones((shape(XFeaScaFinal)[1], 1))
    obj1.fminCg(theta, XFeaScaFinal, obj1.y, obj1.lamda)
    #print "obj1.res[0]",obj1.res[0]
    #print "obj1.res[1]",obj1.res[1]
    obj1.showDataFigure(XFeaScaFinal, p, feaAvgArr, maxFeaArr, minFeaArr)
    Xval = obj1.polyFeatures(p, obj1.Xval)
    Jtheta_TrainSets, Jtheta_CvSets = obj1.learnCurvesPoly(theta, X, obj1.y, Xval, obj1.yval)
    ####画出learningcurves
    obj1.showLearnCurves(Jtheta_TrainSets, Jtheta_CvSets)
    print "#@@@@@@@@@@@@@@@@@@@@@@@@@", obj1.res[1]
    print "#@@@@@@@@@@@@@@@@@@@@@@@@@", obj1.res[0]
    obj1.setLamda(0.3)
    print "更改之后的lamda:", obj1.lamda
    obj1.fminCg(theta, XFeaScaFinal, obj1.y, obj1.lamda)  # lamda =1
    obj1.showDataFigure(XFeaScaFinal, p, feaAvgArr, maxFeaArr, minFeaArr)
    Jtheta_TrainSets, Jtheta_CvSets = obj1.learnCurvesPoly(theta, X, obj1.y, Xval, obj1.yval)
    ####画出learningcurves
    obj1.showLearnCurves(Jtheta_TrainSets, Jtheta_CvSets)
    print "#@@@@@@@@@@@@@@@@@@@@@@@@@", obj1.res[1]
    print "#@@@@@@@@@@@@@@@@@@@@@@@@@", obj1.res[0]