__author__ = 'wangxiang'
# -*- coding:utf8 -*-
from math import sqrt, floor, ceil
from numpy import shape, std, asarray, mean, dot, linalg, reshape, random, ones
import matplotlib.cm as CM
from scipy import io as sio
import matplotlib.pyplot as plt


class PCA(object):
    def __init__(self):
        self.data = sio.loadmat("data/pca/ex7data1.mat")
        self.X = self.data['X']
        print "the data  shape is:", shape(self.X)

    def showPic(self):
        n = shape(self.X)[0]
        print "记录条数：", n
        xcord = [];
        ycord = []
        for i in range(n):
            xcord.append(self.X[i, 0])
            ycord.append(self.X[i, 1])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xcord, ycord, s=30, c='red', marker='s')
        # plt.xlabel('Population of City in 10,000s')
        #plt.ylabel('Profit in $10,000s')
        plt.show()

    def featureScaling(self):
        '''
        m = shape(self.X)[0]  # m代表记录条数
        n = shape(self.X)[1]  # n 代表特征数目
        avgFea = []
        stdFea = []
        for i in range(n):
            avgFea.append(sum(self.X[:, i]) * 1.0 / m)
            #maxFea.append(max(dataArr2[:, i]))
            #minFea.append(min(dataArr2[:, i]))
            stdFea.append(std(self.X[:,i]))
        # print avgFea,maxFea,minFea,shape(array(avgFea))
        #print array(avgFea),array(maxFea)
        feaAvgArr = asarray(avgFea)
        stdFea=asarray(stdFea)
        print feaAvgArr
        print stdFea
        '''
        avgFea = mean(self.X, axis=0)
        stdFea = std(self.X, axis=0, ddof=1)  # 无偏估计  分母为n-1
        print avgFea
        print stdFea
        self.xFScal = (self.X - avgFea) / stdFea  # featureScaling 得到的数据
        print "feature scaing 之后的数据为", self.xFScal
        # dataFeaScalMat = (dataArr2 - feaAvgArr) / (maxFeaArr - minFeaArr)

    def runPCA(self):
        self.conMartix = dot(self.X.T, self.X) / shape(self.X)[0]
        self.U, self.S, self.V = linalg.svd(self.conMartix)  # 奇异值分解求特征向量，特征值，
        # print "U",U
        #print "S",S
        #print "V",V

        #print "############",shape(self.conMartix)
        #print "#############",self.conMartix

    def projectData(self, K):
        self.pcaXFScal = dot(self.U[:, K - 1:K].T, self.xFScal.T)
        # print self.pcaXFScal
        #print shape(self.pcaXFScal)
        #print type(self.pcaXFScal)
        self.pcaXFScal = reshape(self.pcaXFScal, (shape(self.xFScal)[0], K))
        print "PCA 之后的数据:", self.pcaXFScal

    def recoverData(self, K):
        print "U:", self.U
        self.approximateXFScal = dot(self.pcaXFScal, self.U[:, K - 1:K].T)  # 恢复近似的featurescaling数据
        print "恢复近似的featurescaling数据", self.approximateXFScal

    def loadFaceImages(self):
        self.data = sio.loadmat("data/pca/ex7faces.mat")
        self.faceX = self.data['X']
        print "the data  shape is:", shape(self.faceX)

    def showFacesFigure(self, dataArr, *args):
        m, n = shape(dataArr)
        print m, n
        example_width = round(sqrt(n))
        print "特征数目：", n
        print "example_width:", example_width  # 20
        example_height = n / example_width  # 20
        display_rows = floor(sqrt(m))  # 10
        display_cols = ceil(m / display_rows)  # 10
        pad = 1
        display_array = -ones((pad + display_rows * (example_height + pad), pad + display_cols * (example_width + pad)))
        print "display_array 的维度为:", shape(display_array)
        curr_ex = 0;
        for j in range(0, int(display_rows)):
            for i in range(0, int(display_cols)):
                # for j in range(0, int(1)):
                #for i in range(0, int(1)):
                if curr_ex >= m:
                    break
                max_val = max(abs(display_array[curr_ex, :]))
                #print "##################maxval",max_val
                display_array[
                int(pad + j * (example_height + pad)): int(pad + j * (example_height + pad)) + int(example_height),
                int(pad + i * (example_width + pad)):int(pad + i * (example_width + pad)) + int(example_width)] = \
                    reshape(dataArr[curr_ex, :], (example_height, example_width)) / max_val
                curr_ex = curr_ex + 1;
            if curr_ex >= m:
                break;
        plt.imshow(display_array.T, CM.gray)  # 类似matlib中的imagesc
        # scaledimage.scaledimage(display_array)
        plt.show()


if __name__ == "__main__":
    obj1 = PCA()
    '''

    #obj1.showPic()
    obj1.featureScaling()
    obj1.runPCA()
    K=1
    obj1.projectData(K)
    obj1.recoverData(K)
    '''
    # load faceImages
    obj1.loadFaceImages()
    # obj1.showFacesFigure(random.permutation(obj1.faceX)[0:100, :])  # 随机取出100条数据
    obj1.showFacesFigure(obj1.faceX[0:100, :])  # 直接将nn中的展示图像方法拿过来，有点不对，但是重点不在这里
    # 接下来都一样五步走




