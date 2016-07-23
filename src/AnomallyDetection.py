__author__ = 'wangxiang'
# -*- coding:utf8 -*-
from numpy import shape, reshape, square, diag, asarray, linalg, dot, sum, exp, arange, std, count_nonzero
import math
import scipy.io as sio


class AnomalyDetection(object):
    def __init__(self, path):
        self.data = sio.loadmat(path)
        self.X = self.data["X"]
        print shape(self.X)
        self.Xval = self.data["Xval"]
        self.yval = self.data["yval"]
        print shape(self.Xval)
        print "正常点个数:", sum(self.yval == 0)
        print "异常点个数:", sum(self.yval == 1)

    def estimateGaussian(self):
        self.u = reshape(self.X.mean(axis=0), (shape(self.X)[1], 1))
        print "self.u#####################", self.u
        # print shape(self.u)
        #print type(self.u)
        self.sigma2 = reshape(square(std(self.X, axis=0)), (shape(self.X)[1], 1))
        print "self.sigma2###################", self.sigma2

    def multivariateGaussian(self, X):
        self.n = shape(self.u)[0]  # n代表特征的个数
        print self.n
        print shape(self.sigma2)[1]
        print shape(self.sigma2)
        if (shape(self.sigma2)[0] == 1 or shape(self.sigma2)[1] == 1):
            self.sigma2Matrix = diag(self.sigma2.flatten())
        self.X_u = X - self.u.flatten()  # X-u

        self.p = math.pow(2 * math.pi, -self.n / 2.0) * math.pow(linalg.det(self.sigma2Matrix), -1 / 2.0) * exp(
            (-1 / 2.0)
            * (sum(dot(self.X_u, linalg.inv(self.sigma2Matrix)) * self.X_u, axis=1)))  # 多元概率密度函数

    def selectThreshold(self):
        self.pmax = max(self.p)
        self.pmin = min(self.p)
        print "self.pmax$$$$$$$$$$$$$", self.pmax
        print "self.pmin$$$$$$$$$$$$$", self.pmin
        self.step = (self.pmax - self.pmin) / 1000.0
        print "self.step$$$$$$$$$$$$$$$$", self.step
        self.yvalAnomaly = count_nonzero(self.yval)
        self.yvalNormal = shape(self.yval)[0] - self.yvalAnomaly
        # print "cv中异常点的个数",self.yvalAnomaly
        self.best_F1 = 0
        self.bestEpsilon = 0
        for epsilon in arange(self.pmin, self.pmax, self.step):
            self.res = self.p < epsilon  # true为异常点   false为正常点
            self.res = reshape(self.res, (shape(self.res)[0], 1))
            resAnomalyNum = count_nonzero(self.res)  # cv中根据现有epsilon 得到的异常点个数
            print "!!!!!!!!!!!!!!!!!!!!!!!!异常点的个数:", resAnomalyNum
            if resAnomalyNum != 0:
                resNormalNum = shape(self.res)[0] - resAnomalyNum
                tp = count_nonzero((self.res == self.yval) & (self.yval == 1))
                tn = count_nonzero((self.res == self.yval) & (self.yval == 0))
                fp = resAnomalyNum - tp
                fn = resNormalNum - tn
                precision = tp * 1.0 / resAnomalyNum
                recall = tp * 1.0 / self.yvalAnomaly
                self.F1_score = 2 * precision * recall / (precision + recall)
            else:
                self.F1_score = 0
                print "异常点个数为0"
            if self.F1_score > self.best_F1:
                self.best_F1 = self.F1_score
                self.bestEpsilon = epsilon
        print "最终的F1 score:", self.best_F1
        print "最终的epsilon为:", self.bestEpsilon

    def getTraingSetAnomaly(self):
        self.multivariateGaussian(self.X)
        self.trainRes = self.p < self.bestEpsilon
        print "训练集中异常点个数:", count_nonzero(self.trainRes)


if __name__ == "__main__":
    path = "data/ad/ex8data1.mat"
    ad = AnomalyDetection(path)
    ad.estimateGaussian()
    ad.multivariateGaussian(ad.X)  # 这一步ng中是为了图形化展示，在异常检测中没什么用
    ad.multivariateGaussian(ad.Xval)
    ad.selectThreshold()
    ad.getTraingSetAnomaly()

    # ##################更复杂的数据#########################
    path = "data/ad/ex8data2.mat"
    ad = AnomalyDetection(path)
    ad.estimateGaussian()
    ad.multivariateGaussian(ad.Xval)
    ad.selectThreshold()
    ad.getTraingSetAnomaly()
