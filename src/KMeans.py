__author__ = 'wangxiang'
# -*- coding:utf8 -*-
import scipy.io as sio
import sys
from numpy import shape, arange, random, argmin, dot, asarray, reshape
import cv2
import scipy.io as sio


class Kmeans(object):
    def __init__(self):
        data = sio.loadmat("data/kmeans/ex7data2.mat")
        self.X = data['X']
        print "###################", self.X
        # print self.X
        #print type(self.X)
        #print len(self.X)
        #print shape(self.X)

    def randomPick(self, K, X):
        # self.initial_centroids=random.sample(self.X,K)
        #print self.initial_centroids
        #print type(self.initial_centroids)
        #print arange(len(self.X)),type(arange(len(self.X)))
        self.initial_centroids = X[random.choice(len(X), K), :]
        self.new_centroids = self.initial_centroids
        #print self.initial_centroids

    def findClosestCentroids(self, X):
        self.C = asarray([argmin([dot(x_i - y_i, x_i - y_i) for y_i in self.new_centroids]) for x_i in X])
        # [np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X]
        # print self.C
        #print type(self.C)

    def computeCentroids(self, K, X):
        self.new_centroids = asarray([X[self.C == k].mean(axis=0) for k in range(K)])
        # print self.new_centroids
        #res=[]
        #for k in range(K):
        #    cc=self.X[self.C==k]
        ##    res.append(cc.mean(axis=0))
        #print res

    def loadPic(self):
        picData = cv2.imread("data/kmeans/bird_small.png")
        print picData, type(picData), shape(picData)
        self.picData = picData / 255.0
        print self.picData
        self.picDataMatrix = reshape(self.picData,
                                     (shape(self.picData)[0] * shape(self.picData)[1], shape(self.picData)[2]))
        print self.picDataMatrix, shape(self.picDataMatrix)
        # picData2=sio.loadmat("data/bird_small.mat")
        #print picData2['A'],type(picData2['A']),shape(picData2['A'])

    def showPic(self):
        X_recovered = self.new_centroids[self.C]
        X_recovered = reshape(X_recovered, (shape(self.picData)[0], shape(self.picData)[1], shape(self.picData)[2]))
        print shape(X_recovered)
        print X_recovered


if __name__ == "__main__":
    obj = Kmeans()
    # print obj.X
    K = 3
    max_iters = 10
    X = obj.X
    obj.randomPick(K, X)
    for i in range(max_iters):
        obj.findClosestCentroids(X)
        obj.computeCentroids(K, X)
    print obj.new_centroids
    print obj.C
    obj.loadPic()
    X = obj.picDataMatrix
    K = 16
    max_iters = 10
    obj.randomPick(K, X)
    print "random pick cluster centroids", obj.new_centroids  # 16个颜色聚类中心
    for i in range(max_iters):
        obj.findClosestCentroids(X)
        obj.computeCentroids(K, X)
    print " ultimate cluster centroids", obj.new_centroids
    print type(obj.new_centroids), shape(obj.new_centroids)
    print obj.C
    print type(obj.C), shape(obj.C)
    obj.showPic()
