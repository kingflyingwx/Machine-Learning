# -*- coding:utf8 -*-
# Training: ME_train.py TrainingDataFile ModelFile

import sys
import os
import math

"""
train-data中80%为训练集，20%为测试集
"""
TrainingDataFile = "./datacleaning/ml.train"
ModelFile = "me.model"
DocList = []
WordDic = {}
FeaClassTable = {}
FeaWeights = {}
ClassList = []
C = 100
MaxIteration = 1000
LogLLDiff = 0.1
CommonFeaID = 10000001


def Dedup(items):
    tempDic = {}
    for item in items:
        if item not in tempDic:
            tempDic[item] = True
    return tempDic.keys()


"""
     DocList:(newDoc,classid) 存储的是每一篇文章的term(按照文中出现的次序)以及该篇文章的classid
     ClassList：总共的各个类别比如1,2,3，就有三类
     WordDic：字典类型,所有训练集中的term，值都是1，最后多加了一个term CommonFeaID 比maxwid大1
     maxwid：所有term中最大的那个term
"""


def LoadData():
    global CommonFeaID
    i = 0
    infile = file(TrainingDataFile, 'r')
    sline = infile.readline().strip()
    maxwid = 0
    while len(sline) > 0:
        pos = sline.find("#")
        if pos > 0:
            sline = sline[:pos].strip()
        words = sline.split(' ')
        if len(words) < 1:
            print ("Format error!")
            break
        classid = int(words[0])
        if classid not in ClassList:
            ClassList.append(classid)
        words = words[1:]
        #remove duplicate words, binary distribution
        words = Dedup(words)
        newDoc = {}
        for word in words:
            if len(word) < 1:
                continue
            wid = int(word)
            if wid > maxwid:
                maxwid = wid  #maxwid存储的是训练集中单词对应数字最大值
            if wid not in WordDic:
                WordDic[wid] = 1
            if wid not in newDoc:
                newDoc[wid] = 1
        i += 1
        DocList.append((newDoc, classid))
        sline = infile.readline().strip()
    infile.close()
    print (len(DocList), "instances loaded!")  #训练集中的文章数
    print (len(ClassList), "classes!", len(WordDic), "words!")
    CommonFeaID = maxwid + 1
    print ("Max wid:", maxwid)
    WordDic[CommonFeaID] = 1


"""
定义好特征函数
FeaClassTable ={term1:({classid:n1，classid2：n2}，{}),term2:({classid:n3},{}),...}
"""


def ComputeFeaEmpDistribution():
    global C
    global FeaClassTable
    FeaClassTable = {}
    for wid in WordDic.keys():
        temppair = ({}, {})
        FeaClassTable[wid] = temppair
    maxCount = 0
    for doc in DocList:
        if len(doc[0]) > maxCount:  #doc为每一篇文章，doc[0]为该片文章的term列表，doc[0]为字典类型
            maxCount = len(doc[0])

    C = maxCount + 1
    for doc in DocList:
        doc[0][CommonFeaID] = C - len(doc[0])
        for wid in doc[0].keys():
            if doc[1] not in FeaClassTable[wid][0]:
                FeaClassTable[wid][0][doc[1]] = doc[0][wid]
            else:
                FeaClassTable[wid][0][doc[1]] += doc[0][wid]

    return


"""
classProbs：类型为list，大小为分类数。 每一篇文章一更新，对应该篇文章各个类的P（y|x）
FeaWeights: 全局变量,格式为term，{1：0.x，2:0.y,3:0.0}},对应所有训练数据集中所有的term在各个类的参数大小 wi，
            一个term的参数对应到每个类都有取值
FeaClassTable：全局变量,格式为{termi:({},{})},元组中第一个字典记录的是各个term（所有训练数据）在训练数据集中出现的次数，对应到f(x,y)，其实就是Ep~(f),因为p~(x,y),x与y同时发生的概率就是x在训练集中的次数/文档数，对一个term以及一个类别来说是常量

               元组中第二个字典是Ep(f)，公式中P~(x)跟P~(x,y)是一样的（文档是等概率的），没次迭代其值都要初始化

"""


def GIS():
    global C
    global FeaWeights
    for wid in WordDic.keys():
        FeaWeights[wid] = {}
        for classid in ClassList:
            FeaWeights[wid][classid] = 0.0
    n = 0
    prelogllh = -1000000.0
    logllh = -10000.0
    while logllh - prelogllh >= LogLLDiff and n < MaxIteration:  #收敛条件
        n += 1
        prelogllh = logllh
        logllh = 0.0
        print ("Iteration", n)

        for wid in WordDic.keys():
            for classid in ClassList:
                FeaClassTable[wid][1][classid] = 0.0
        for doc in DocList:
            classProbs = [0.0] * len(ClassList)
            sum = 0.0
            for i in range(len(ClassList)):
                classid = ClassList[i]
                pyx = 0.0
                for wid in doc[0].keys():
                    pyx += FeaWeights[wid][classid]

                pyx = math.exp(pyx)
                classProbs[i] = pyx
                sum += pyx
            for i in range(len(ClassList)):
                classProbs[i] = classProbs[i] / sum
            for i in range(len(ClassList)):
                classid = ClassList[i]
                if classid == doc[1]:
                    logllh += math.log(classProbs[i])
                for wid in doc[0].keys():
                    FeaClassTable[wid][1][classid] += classProbs[i] * doc[0][wid]
                    #update feature weights
        for wid in WordDic.keys():
            for classid in ClassList:
                empValue = 0.0
                if classid in FeaClassTable[wid][0]:
                    empValue = FeaClassTable[wid][0][classid]
                modelValue = 0.0
                if classid in FeaClassTable[wid][1]:
                    modelValue = FeaClassTable[wid][1][classid]
                if empValue == 0.0 or modelValue == 0.0:
                    continue
                FeaWeights[wid][classid] += math.log(
                    FeaClassTable[wid][0][classid] / FeaClassTable[wid][1][classid]) / C
        print ("Loglikelihood:", logllh)
    return


def SaveModel():
    outfile = file(ModelFile, 'w')
    for wid in FeaWeights.keys():
        outfile.write(str(wid))
        outfile.write(' ')
        for classid in FeaWeights[wid]:
            outfile.write(str(classid))
            outfile.write(' ')
            outfile.write(str(FeaWeights[wid][classid]))
            outfile.write(' ')
        outfile.write('\n')
    outfile.close()


#main framework
if __name__ == "__main__":
    print "start training:"
    # TrainingDataFile = sys.argv[1]
    # ModelFile = sys.argv[2]
    LoadData()
    ComputeFeaEmpDistribution()
    GIS()
    SaveModel()
    print "end training"

