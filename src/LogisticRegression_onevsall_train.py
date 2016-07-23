__author__ = 'wangxiang'
# -*- coding:utf8 -*-
import pickle
import matplotlib.pyplot as plt
from numpy.ma import exp
from scipy import io as sio
import time
import LogisticPic
from numpy import shape, ma, random, ones, reshape, arange, asarray, zeros, concatenate, dot
from scipy.optimize import fmin_bfgs, minimize, fmin_cg
import matplotlib.cm as CM


def loadDataSet(path):
    data = sio.loadmat(path)
    return data['X'], data['y']


def showDataFigure(dataArr, *args):
    m, n = shape(dataArr)
    # print m, n
    example_width = round(ma.sqrt(n))
    # print "特征数目：", n
    #print "example_width:", example_width  # 20
    example_height = n / example_width  # 20
    display_rows = ma.floor(ma.sqrt(m))  # 10
    display_cols = ma.ceil(m / display_rows)  # 10
    pad = 1
    display_array = -ones((pad + display_rows * (example_height + pad), pad + display_cols * (example_width + pad)))
    #print "display_array 的维度为:", shape(display_array)
    curr_ex = 0;
    for j in range(0, int(display_rows)):
        for i in range(0, int(display_cols)):
            #for j in range(0, int(1)):
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
    #scaledimage.scaledimage(display_array)
    plt.show()


'''
    x1cord = [];y1cord=[]
    x2cord = [];y2cord=[]
    #print type(labelMat[1])
    for i in range(n):
        if labelMat[i]==[1]:
            x1cord.append(dataArr[i,1])
            y1cord.append(dataArr[i,2])
        else:
            x2cord.append(dataArr[i,1])
            y2cord.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    admitted=ax.scatter(x1cord, y1cord, s=30, c='black', marker='+')
    nadmit=ax.scatter(x2cord,y2cord,s=30,c='yellow',marker='o')
    plt.legend((admitted,nadmit),('admitter','not admitted'), scatterpoints=1)  # 加图例
    if args:   # 非None（空） 即为真
        x = arange(30, 100, 5)
        y = -args[0]*1.0/args[2] - args[1]*x*1.0/args[2]
        ax.plot(x, y)
    plt.xlabel('Exam1 Score')
    plt.ylabel('Exam2 Score')
    plt.show()
    '''


def sigmodFunction(dataArr, theta):
    z = dot(dataArr, theta)
    p = 1e-5
    hx = 1 / (1 + exp(-z))
    # print "sigmod Function 中hx 的值为:",hx
    hx_new = []
    for f in hx.flat:
        if f == 1:
            hx_new.append([f - p])
        elif f == 0:
            hx_new.append([f + p])
        else:
            hx_new.append([f])
    hx = asarray(hx_new)
    return hx


def getGradient_Regular(theta, *args):
    # print "getGradient2 中theta为：",theta
    dataArr = asarray(args[0])
    labelArr = asarray(args[1])
    lamda = args[2]
    m, n = shape(dataArr)
    X = dataArr.transpose()
    theta = reshape(theta, (n, 1))
    hx = sigmodFunction(dataArr, theta)
    hx_minux_y = hx - labelArr
    gra = dot(X, hx_minux_y) * 1.0 / m  # dot(array(dataMat).transpose(),hx-array(labenMat))*1.0/m
    # print "!!!!!!!!!!!!!!!!!!转变之前的gra",gra.flatten()
    gra = gra + lamda * 1.0 * theta / m
    gra[0, 0] = dot(X[0, :], hx_minux_y) * 1.0 / m
    # print "!!!!!!!!!!!!转变之后的gra",gra.flatten()
    return gra.flatten()


def costFunction_Regular(theta, *args):
    print "现在在调用正则化的cost函数"
    # print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",shape(theta)
    dataArr = asarray(args[0])  # args0 为特征数据  args1 为类别数据
    labelArr = asarray(args[1])
    m, n = shape(dataArr)  # dataArr 已经加入bias
    # print "记录条数:",m
    theta = reshape(theta, (n, 1))
    #print "costFunction_Regular中theta的值为：", theta
    hx = sigmodFunction(dataArr, theta)  #计算预测的类别概率值
    loghx = ma.log(hx)
    #print "log hx:",loghx
    yhx = dot(loghx.transpose(), labelArr)
    #print "yhx:",yhx
    log1_hx = ma.log(1 - hx)
    #print "log1_hx:",log1_hx
    #print (-yhx-dot(log1_hx.transpose(),(1-labelArr)))*1.0/m
    #print "lameda 值为:", args[2]  # args[2]为传入的lameda参数
    jtheta = (-yhx - dot(log1_hx.transpose(), (1 - labelArr))) * 1.0 / m + args[2] * 1.0 / (2 * m) * (
        dot(theta.transpose(), theta) - theta[0, 0] ** 2)
    #gra=getGradient(dataArr,labelArr,theta,m)
    #print type(jtheta),type(gra.flatten())
    #print gra
    #print "###################",type(array(jtheta)[0]),array(jtheta)[0]
    #print "costFunction_Regular得到的jtheta值为:", type(jtheta.flatten()[0]), jtheta, jtheta.flatten()[0]
    print "&&&&&7jtheta.flatten()[0]", jtheta.flatten()[0]
    return jtheta.flatten()[0]


def predictRes(x, theta, y, threshold):
    dataArr = asarray(x)
    labelArr = asarray(y)
    n = shape(y)[0]
    # print theta
    z = dot(x, theta)
    p = 1.0 / (1 + exp(-1 * z))
    p[p < threshold] = 0
    p[p >= threshold] = 1
    # print p
    res = p - labelArr
    count = 0
    for r in res.flat:
        if r != 0:
            count += 1
    print count
    print n
    return (n - count * 1.0) / n


if __name__ == "__main__":
    print "begin logistic regression one vs all"
    timeBeg = time.time()
    path = "data/logreg/handwritten_digit.mat"
    dataX, dataY = loadDataSet(path)
    m, n = shape(dataX)
    # print dataY
    showDataFigure(random.permutation(dataX)[0:100, :])  # 随机取出100条数据
    dataX = concatenate((ones((m, 1)), dataX), axis=1)
    K = 10  # 类别个数
    theta = zeros((K, n + 1))
    lamda = 1
    res = []
    for i in arange(1, K + 1):
        print i
        dataY_new = dataY == i
        xopt = fmin_cg(costFunction_Regular, theta[i - 1, :], getGradient_Regular, (dataX, dataY_new, lamda),
                       maxiter=400)
        res.append(xopt)
    print "耗时：", time.time() - timeBeg
    print time.strftime("%Y%m%d %H:%M:%S", time.localtime(time.time()))
    f = open("data/logistic_onevsall_model_fminbfgs", "w")
    res_bin = pickle.dumps(res, 1)
    pickle.dump(res_bin, f)
    f.close()
    '''
    #########################################################################################################################
    ##################################带有regularized的logistic regression 算法#########################
    path="data/logistic_data2.data"
    dataMat,labelMat=loadDataSet(path)
    #showDataFigure(dataMat,labelMat)
    degree=6
    dataNew=mapFeature(dataMat,degree)
    print "^^^^^^^^^^^^^^",dataNew

    theta=zeros((28,1)).flatten()
    #jtheta=costFunction_Regular(theta,*(dataNew,labelMat,1))
    #print jtheta
    lamda=1
    xopt=fmin_bfgs(costFunction_Regular,theta,fprime=getGradient_Regular,args=(dataNew,labelMat,lamda),maxiter=400,full_output=1,retall=0)
    print xopt[0],xopt[1]
    LogisticPic.showDataFigure(xopt[0],dataMat,labelMat,lamda)
    '''

