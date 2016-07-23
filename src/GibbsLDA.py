# -*- coding:utf8 -*-
import sys
import os
import random
import math

reload(sys)
sys.setdefaultencoding("utf-8")

inpath = "./train-data"
outFile = "_lda_out_"
DocList = []
DocNameList = []
WordDic = {}
WordList = []

nIteration = 3
# Epsilon = 0.2
K = 3
N = 10000
W = 10000
Alpha = 0.1
Beta = 0.1
TotalWordCount = 0
nwz = []
nzd = []
nz = [0] * K
nd = []
pwz = []
pdz = []
"""
   基于gibbs lda算法，针对所有文章的每一个word进行计算，对比 基于em glsa算法
   DocList((widlist,zlist)): widlist为一个列表，保存的每一篇文章中顺序word的下标，zlist为一个列表，保存的是每一篇文章中相应word对应的主题Z，全都为-1
   plsa与lda DocList里面存储的内容有区别，plsa存储的是每一篇文章的term以及对应次数，而gibbs lda存储的是每一篇文章的word以及对应的主题，所有的
word都列了出来
   N：文章数
   W: 所有训练集文章中的term 数
   N,W都是有顺序的，按照训练集中文档以及文章中各个word的先后顺序来的
   nwz：K*W维度
   nzd：N*K维度
   nd：列表，每一个元素为一篇文章中的word数
"""
def LoadData():
    global nwz
    global nzd
    global nd
    global N
    global W
    global TotalWordCount
    i = 0
    nd = []
    for filename in os.listdir(inpath):
        if filename.find(".txt") == -1:
            continue
        i += 1
        infile = file(inpath + '/' + filename, 'r')
        DocNameList.append(filename)
        content = infile.read().strip()
        content = content.decode("utf-8")
        words = content.replace('\n', ' ').split(' ')
        # print  str(len(words)) + "\t" + filename
        widlist = []
        zlist = []
        wordNum = 0
        for word in words:
            if len(word.strip()) < 1:
                continue
            if word not in WordDic:
                WordDic[word] = len(WordList)
                WordList.append(word)
            wid = WordDic[word]
            wordNum += 1
            widlist.append(wid)
            zlist.append(-1)
        DocList.append((widlist, zlist))
        nd.append(wordNum)
        TotalWordCount += wordNum
    N = len(DocList)
    W = len(WordList)
    nwz = [[]] * K
    nzd = [[]] * N
    for i in range(K):
        nwz[i] = [0] * W
    for i in range(N):
        nzd[i] = [0] * K  #对应全部训练接 维度N*K
    print len(DocList), "files loaded!"
    print len(WordList), "unique words in total!"
    print TotalWordCount, "occurrences in total!"


"""
 DocList((widlist,zlist)): widlist为一个列表，保存的每一篇文章中word的下标，zlist为一个列表，保存的是每一篇文章中相应word对应的初始随机主题Z(0~K-1)
 nwz:K*W,整个训练集每一个主题每一个term出现的数目（即word数）
 nzd:N*K,整个训练集每一个文章每一个主题对应的word数目
 nz ：K维 ，整个训练集中每个主题出现的word数目
"""


def Init():
    global nwz
    global nzd
    global nz
    did = 0
    for doc in DocList:  #循环遍历训练集中每一篇文
        for j in range(len(doc[1])):
            tempz = random.randint(0, K - 1)
            doc[1][j] = tempz
            nwz[tempz][doc[0][j]] += 1
            nzd[did][tempz] += 1
            nz[tempz] += 1
        did += 1
    print "Init over!"


def Sample(i, j):  #i为第i个文档，j为第j个word
    global nwz
    global nzd
    global nz
    #remove zi from the counts
    z = DocList[i][1][j]
    w = DocList[i][0][j]
    nwz[z][w] -= 1
    nzd[i][z] -= 1
    nz[z] -= 1
    Wbeta = W * Beta
    Kalpha = K * Alpha
    #compute cumulative probability
    p = [0.0] * K
    for k in range(K):
        p[k] = (nwz[k][w] + Beta) * (nzd[i][k] + Alpha) / (
        (nz[k] + Wbeta) * (nd[i] + Kalpha))  #Gibbs updating rule  计算第i篇文档w word的第k个主题的概率
        if k > 0:
            p[k] += p[k - 1]
    #sampling
    randomValue = random.random() * p[K - 1]
    newz = -1
    for k in range(K):
        if p[k] > randomValue:
            newz = k
            break
    #update counts
    DocList[i][1][j] = newz
    nwz[newz][w] += 1
    nzd[i][newz] += 1
    nz[newz] += 1
    return math.log(p[K - 1])


def ComputePP():
    pp = 0.0
    did = 0
    Wbeta = W * Beta  # Beat初始值 0.1；Alpha初始值0.1
    Kalpha = K * Alpha
    for doc in DocList:
        for j in range(len(doc[1])):
            z = doc[1][j]
            w = doc[0][j]
            probSum = 0.0
            for k in range(K):
                #nwz
                probSum += (nwz[k][w] + Beta) * (nzd[did][k] + Alpha) / (
                (nz[k] + Wbeta) * (nd[did] + Kalpha))  #Gibbs updating rule

            pp += math.log(probSum)
        did += 1
    pp /= TotalWordCount
    #pp = math.exp(-pp)
    return pp


def Learn():
    prepp = -1
    n = 0
    pp = ComputePP()
    print n, "iteration:", pp
    while n < nIteration:  # and math.fabs(prepp-pp) > Epsilon:
        did = 0
        for doc in DocList:
            for j in range(len(doc[1])):
                Sample(did, j)
            did += 1
        prepp = pp
        pp = ComputePP()
        n += 1
        print n, "iteration:", pp
    print "learning finished!"


"""
最终得到的pwz以及pzd 中的z都是一样的，序号也是对应的，整个训练过程中都是严格按照z序号对应的
即w中的topic 0 跟d中的topic0 是一样的
"""


def EstimateParams():
    global pwz
    global pdz
    pwz = [[]] * K
    pdz = [[]] * K
    for i in range(K):
        pwz[i] = [0.0] * W
        sum = 0.0
        for j in range(W):  #每个词项
            sum += nwz[i][j]
        for j in range(W):
            pwz[i][j] = nwz[i][j] / sum
    # for i in range(K):
    #   pdz[i] = [0.0]*N
    #   sum = 0.0
    #   for j in range(N):
    #     sum += nzd[j][i]
    #   for j in range(N):
    #     pdz[i][j] = nzd[j][i] / sum
    #我写的code
    for i in range(K):
        pdz[i] = [0.0] * N
    for j in range(N):
        sum = 0.0
        for i in range(K):
            sum += nzd[j][i]
        for i in range(K):
            pdz[i][j] = nzd[j][i] / sum


#main framework
print "load data"
LoadData()
print "init"
Init()
print "learn"
Learn()
print "estimate"
EstimateParams()

print "save model"
#write the pwz pdz into outFile
TopN = 20
pwzfile = file(outFile + "pwz.txt", 'w')
pdzfile = file(outFile + "pdz.txt", 'w')
for i in range(K):
    templist = []
    for j in range(W):
        templist.append((j, pwz[i][j]))
    templist.sort(key=lambda x: x[1], reverse=True)
    pwzfile.write("Topic " + str(i))
    pwzfile.write("\n")
    j = 0
    while j < W and j < TopN:
        pwzfile.write("\t" + WordList[templist[j][0]])
        pwzfile.write("\n")
        j += 1
for i in range(K):
    templist = []
    for j in range(N):
        templist.append((j, pdz[i][j]))
    templist.sort(key=lambda x: x[1], reverse=True)
    pdzfile.write("Topic " + str(i))
    pdzfile.write("\n")
    j = 0
    while j < N and j < TopN:
        pdzfile.write("\t" + DocNameList[templist[j][0]])
        pdzfile.write("\n")
        j += 1
pwzfile.close()
pdzfile.close()
