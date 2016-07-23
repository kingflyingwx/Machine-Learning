__author__ = 'wangxiang'
# -*- coding:utf8 -*-
from numpy import loadtxt, shape
import urllib
import time
import re
import stemmer
from  numpy import zeros
from scipy import io as sio
from  SVM import SVM


class ProcessSpam:
    def __init__(self):
        '语料库一般10000到50000个单词'
        print "初始化。。。"

    def loadMailData(self):
        f = open("data/svm/spamSample1.txt", "r")
        self.data = f.read()
        # print "data is :",self.data
        self.dataLower = self.data.lower()
        #print "dataLower is :",self.dataLower
        f.close()

    def loadVocabList(self):
        f = open("data/svm/vocab.txt")
        self.vocabList = []
        for line in f:
            self.vocabList.append(line.split()[1])
        f.close()

    def proMailData(self):
        pattern = re.compile("<[^<>]+>")
        self.dataMailFinal = pattern.sub(" ", self.dataLower)  # 将所有的html标签替换成空格
        # print self.dataMailFinal
        pattern = re.compile("[0-9]+")
        self.dataMailFinal = pattern.sub("number", self.dataMailFinal)
        pattern = re.compile(r"(http|https)://[^\s]*")
        self.dataMailFinal = pattern.sub("httpaddr", self.dataMailFinal)
        pattern = re.compile(r"[\S]+@[\S]+")
        self.dataMailFinal = pattern.sub("emailaddr", self.dataMailFinal)
        pattern = re.compile("[$]+")
        self.dataMailFinal = pattern.sub("dollar", self.dataMailFinal)
        #print self.dataMailFinal
        #self.dataMailSet=re.split("[(){%}:\-#\'!/$'@\n\r?><_,.\" ]",self.dataMailFinal)
        self.dataMailSet = re.split("\W", self.dataMailFinal)
        #print self.dataMailSet
        #print len(self.dataMailSet[0])
        #print len(self.dataMailSet)
        self.dataMailSet = filter(lambda x: len(x) >= 1, self.dataMailSet)
        print "处理好之后邮件中各个词为:", self.dataMailSet

    def porterStemmer(self):
        ps = stemmer.Stemmer()
        self.dataMailStem = []
        for d in self.dataMailSet:
            try:
                self.dataMailStem.append(ps.stem(d))
            except BaseException, Argument:
                print "error", Argument

        print "波特词干提取之后：", self.dataMailStem

    def getWordIndices(self):
        self.wordIndices = []
        for d in self.dataMailStem:
            try:
                self.wordIndices.append(self.vocabList.index(d))
            except BaseException, Arguments:
                print "error", Arguments

    def getFeatures(self):
        self.n = len(set(self.wordIndices))
        self.nVol = len(self.vocabList)
        self.mailFeatures = zeros((self.nVol, 1))
        # print type(self.mailFeatures)
        #print shape(self.mailFeatures)
        #print self.mailFeatures[1500:2000,0]
        for d in set(self.wordIndices):
            self.mailFeatures[d, 0] = 1  # 特征向量
            #print self.mailFeatures[1500:2000,0]
            ##def loadMailDataSet(self,path):
            #    self.mailDataSet=sio.loadmat(path)
            #    print self.mailDataSet
            #    print type(self.mailDataSet)
            #   self.mailX=self.mailDataSet['X']
            #    self.mailY=self.mailDataSet['y']
            #    print self.mailX
            #    print self.mailY


if __name__ == "__main__":
    time_ben = time.time()
    obj1 = ProcessSpam()
    # 加载邮件数据
    obj1.loadMailData()
    #加载词汇列表
    obj1.loadVocabList()
    '''
    预处理邮件数据
    1) 将邮件中所有的单词统一小写处理
    2）将所有的数字统一变为 ‘number’
    3）将所有的邮件统一变为‘emailaddr’
    4）将所有的$统一变为 ‘dollar’
    5）将所有的url统一变为‘httpaddr’
    6） 将html标签都去掉
    7）将所有非字母数字以及下划线_的符号都去掉，将tab 多个空格 等都变成一个space
     '''
    obj1.proMailData()
    #波特词干提取
    obj1.porterStemmer()
    obj1.getWordIndices()
    print obj1.wordIndices
    #print obj1.wordIndices
    #print len(obj1.wordIndices)
    #print len(set(obj1.wordIndices))
    obj1.getFeatures()
    #print obj1.mailFeatures.T
    print shape(obj1.mailFeatures.T)
    svmObj = SVM("data/svm/spamTrain.mat", "data/svm/spamTest.mat", obj1.mailFeatures.T)
    svmObj.processData()
    c = 100
    t = 0
    svmObj.trainModel(c, t)
    t = 2
    svmObj.trainModel(c, t)
    print "耗费的时间为:", time.time() - time_ben