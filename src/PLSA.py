# -*- coding:utf8 -*-

import sys
import os
import random
import math

inpath = "D://topic_model_data/"
outFile = "D://topic_model_data/res/plsaout"
DocList = []
DocNameList = []
WordDic = {}
WordList = []

nIteration = 30  # 30
Epsilon = 0.01
K = 100
N = 10000
W = 10000
TotalWordCount = 0
pwz = []
pzd = []
# pz = [0.0]*K
newpwz = []

"""
    初始化：
    N ： 文章数
    W : term 数 (term采用加入WordList时WordList长度对应，即下标)
    K ：100 为特征数，默认值
    pwz : K*W list
    pzd : N*K list
    TotalWordCount:  训练数据所有文章的word总数
    WordList：全部训练集的词项term
"""


def LoadData():
    global pwz
    global pzd
    global N
    global W
    global TotalWordCount
    i = 0

    for filename in os.listdir(inpath):
        if filename.find(".txt") == -1:
            continue
        i += 1
        infile = file(inpath + '/' + filename, 'r')
        DocNameList.append(filename)
        content = infile.read().strip()
        content = content.decode("utf-8")
        words = content.replace('\n', ' ').split(' ')
        newdic = {}
        widlist = []
        freqlist = []
        wordNum = 0
        for word in words:
            if len(word.strip()) < 1:
                continue
            if word not in WordDic:
                WordDic[word] = len(WordList)
                WordList.append(word)
            wid = WordDic[word]
            wordNum += 1
            if wid not in newdic:
                newdic[wid] = 1.0
            else:
                newdic[wid] += 1.0
        for (wid, freq) in newdic.items():
            widlist.append(wid)
            freqlist.append(freq)
        DocList.append((widlist, freqlist))
        TotalWordCount += wordNum
    N = len(DocList)
    W = len(WordList)
    pwz = [[]] * K
    pzd = [[]] * N
    for i in range(K):
        pwz[i] = [0.0] * W
    for i in range(N):
        pzd[i] = [0.0] * K
    print len(DocList), "files loaded!"
    print len(WordList), "unique words in total!"
    print TotalWordCount, "occurrences in total!"
    print DocNameList, "DocNameList"


"""
初始化参数(也就是最终我们要求的):p(z/d) 维度N*K ,p(w/z) 维度K*W
"""


def Init():
    global pwz
    global pzd
    for i in range(K):
        tempsum = 0.0
        for j in range(W):
            pwz[i][j] = random.random()
            tempsum += pwz[i][j]
        for j in range(W):
            pwz[i][j] /= tempsum
    for i in range(N):
        tempsum = 0.0
        for j in range(K):
            pzd[i][j] = random.random()
            tempsum += pzd[i][j]
        for j in range(K):
            pzd[i][j] /= tempsum  #保证所有的概率之和为1
    print "Init over!"


"""
基于EM算法的PLSA概率潜层语义模型的时间复杂度为O(nN*K*W)(都是稀疏列表，最坏情况的时间复杂度) 小n为迭代次数
                                 空间复杂度为O(N*K+K*W)
                                 计算超慢，有时间用numpy矩阵运算
N篇文章K个主题W个term，pwz K*W  pzd N*K   都有值，取大的作为聚类或者推荐
"""


def EMIterate():
    global pwz
    global newpwz
    global pzd
    global TotalWordCount
    newpwz = [[]] * K  # K*W维度
    for i in range(K):
        newpwz[i] = [0.0] * W
    did = 0
    pp = 0.0
    while did < len(DocList):  # 对每一篇文章进行遍历
        #if did % 100 == 0:
        #    print did,"docs processed!"
        newpzd = [0.0] * K
        doc = DocList[did]  # 获取第did篇文章的信息：（词term列表，词term对应次数列表）   全放在内存里很占用空间

        ndoc = 0
        for index in range(len(doc[0])):
            word = doc[0][index]
            ndw = doc[1][index]
            ndoc += ndw
            pzdw = [0.0] * K
            pzdwsum = 0.0
            # 该for循环针对一篇文章的一个单词计算出所有主题的一个后验概率 pzdw
            for z in range(K):
                pzdw[z] = pzd[did][z] * pwz[z][word]
                pzdwsum += pzdw[z]
            pp += ndw * math.log(pzdwsum)

            for z in range(K):
                pzdw[z] = pzdw[z] / pzdwsum
                temppzdw = ndw * pzdw[z]
                newpwz[z][word] += temppzdw

                newpzd[z] += temppzdw
        #normalize and update pzd
        for z in range(K):
            pzd[did][z] = newpzd[z] / ndoc  #更新pzd  did < len(DocList) 循环一次得到更新后的K个新的pzd
        did += 1
    #normalize and update pwz
    for z in range(K):
        pwzsum = 0.0
        for wid in range(W):
            pwzsum += newpwz[z][wid]
        for wid in range(W):
            newpwz[z][wid] /= pwzsum
    pwz = newpwz  #跟新后的pwz
    pp /= TotalWordCount
    #pp = math.exp(-pp)
    return pp


def Learn():
    prepp = -1
    n = 0
    pp = EMIterate()  #一次EM迭代算法    pp的作用是计算是否收敛，如果两次迭代的结果l（theta）联合分布查的绝对值在0.01 范围之内就算收敛
    print n, "iteration:", pp
    while n < nIteration and math.fabs(prepp - pp) > Epsilon:  # Epsilon 为0.01
        prepp = pp
        pp = EMIterate()
        n += 1
        print n, "iteration:", pp
    print "learning finished!"

#main framework
LoadData()
Init()
Learn()

#write the pwz pdz into outFile
TopN = 20
pwzfile = file(outFile + "pwz.txt", 'w')
pdzfile = file(outFile + "pdz.txt", 'w')
for i in range(K):
    templist = []
    for j in range(W):
        templist.append((j, pwz[i][j]))
    templist.sort(key=lambda x: x[1])
    pwzfile.write("Topic " + str(i))
    pwzfile.write("\n")
    j = 0
    while j < W and j < TopN:
        pwzfile.write("\t" + WordList[templist[j][0]].encode("utf-8"))
        pwzfile.write("\t" + str(templist[j][1]))
        pwzfile.write("\n")
        j += 1
for i in range(K):
    templist = []
    for j in range(N):
        templist.append((j, pzd[j][i]))
    templist.sort(key=lambda x: x[1])
    pdzfile.write("Topic " + str(i))
    pdzfile.write("\n")
    j = 0
    while j < N and j < TopN:
        print "write pdzfile: " + str(j)
        pdzfile.write("\t" + DocNameList[templist[j][0]])
        pdzfile.write("\t" + str(templist[j][1]))
        pdzfile.write("\n")

        j += 1

print "writing finish!"
pwzfile.close()
pdzfile.close()