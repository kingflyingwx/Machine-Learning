__author__ = 'wangxiang'
# -*- coding:utf8 -*-
from scipy.optimize import fmin_cg
import pickle
from numpy import linalg as LA, amax
from numpy import shape, ones, concatenate, dot, asarray, reshape, arange, zeros
from numpy.ma import exp, sqrt, floor, ceil, log, sin, remainder
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.cm as CM
import scipy.io as sio

num = 0


def loadDataSet(path):
    data = sio.loadmat(path)
    return data['X'], data['y']


def loadWeights(path):
    weights = sio.loadmat(path)
    return weights['Theta1'], weights['Theta2']


def sigmodFunction(z2):
    p = 1e-5
    hx = 1.0 / (1 + exp(-z2))
    # print "sigmod Function 中hx 的值为:",hx
    m, n = shape(hx)
    hx_new = []
    for f in hx.flat:
        if f == 1:
            hx_new.append([f - p])
        elif f == 0:
            hx_new.append([f + p])
        else:
            hx_new.append([f])
    hx = asarray(hx_new)
    hx = reshape(hx, (m, n))
    return hx


def predictAccurancy(dataX, theta, input_layer_size, hidden_layer_size, K):
    m1 = hidden_layer_size
    n1 = input_layer_size + 1
    m2 = K
    n2 = hidden_layer_size + 1
    theta1 = reshape(theta[0:m1 * n1], (m1, n1))
    theta2 = reshape(theta[m1 * n1:m1 * n1 + m2 * n2], (m2, n2))
    z2 = dot(dataX, theta1.T)
    a2 = sigmodFunction(z2)
    m, n = shape(a2)
    a2 = concatenate((ones((m, 1)), a2), axis=1)
    z3 = dot(a2, theta2.T)
    a3 = sigmodFunction(z3)
    print "###############预测结果的维度为####", shape(a3)
    print "#################预测结果为##", a3
    print "##################预测结果中的最大值为####", amax(a3, axis=1)
    preClass = a3.argmax(axis=1) + 1
    # print "预测得到的类别为:",reshape(preClass,(5000,1))
    return preClass


def showDataFigure(dataArr, *args):
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


def preProcessDataY(dataY, m, K):
    dataY_new = zeros((m, K))
    for i in arange(m).flat:
        dataY_new[i, dataY[i] - 1] = 1  # 将每一个样本的类别值映射成 1*K 的向量。   类别1 对应下标0， 类别10 对应下标9
    return dataY_new


def getPredictRes(dataX, theta1, theta2):
    z2 = dot(dataX, theta1.T)
    a2 = sigmodFunction(z2)
    m, n = shape(a2)
    a2 = concatenate((ones((m, 1)), a2), axis=1)
    z3 = dot(a2, theta2.T)
    a3 = sigmodFunction(z3)
    # print "@##########################################################################",a3
    return a3


def getRegularTerm(theta1, theta2, lamda, m):
    # theta1_square=dot(theta1.flatten()[1:],(theta1.flatten()[1:]).T)
    #theta2_square=dot(theta2.flatten()[1:],theta2.flatten()[1:].T)
    theta1_square = dot(theta1[:, 1:].flatten(), (theta1[:, 1:].flatten()).T)
    theta2_square = dot(theta2[:, 1:].flatten(), theta2[:, 1:].flatten().T)
    #print "@#@#@#@#@#@#@#@#@#",theta1_square,theta2_square
    return lamda * 1.0 * (theta1_square + theta2_square) / (2 * m)


def costFunction_Regular(theta, *args):
    # 参数顺序theta1,theta2 ,dataX_new,dataY_new
    #theta=reshape(theta,())
    '''
    theta1=args[0]   #传入初始theta的目的就是为了获得他们的维度而已
    theta2=args[1]
    dataX=args[2]  #5000*401
    m=shape(dataX)[0]
    dataY=args[3]  #5000*10
    K=args[4]   #   类别个数
    lamda =args[5]
    #print "dataX的维度:",shape(dataX)
    #print "dataY的维度:",shape(dataY)
    m1,n1=shape(theta1)
    m2,n2=shape(theta2)
    #print m1,n1,m1*n1
    theta1=reshape(theta[0:m1*n1],(m1,n1))
    theta2=reshape(theta[m1*n1:m1*n1+m2*n2],(m2,n2))
    '''
    theta1, theta2, dataX, K, dataY, m = getArgs(theta, args)
    hx = getPredictRes(dataX, theta1, theta2)  # 5000*10
    #print "!@!@!@!@!@!!@!@!@!@!@!@!@@!,",shape(hx)
    jtheta = 0
    '''
    K为类别数目，m为训练样本数
    之所以这里用了一个for循环而不用向量相乘，是因为如果纯用向量相乘最终只有对角线的数据有用，如果m，n很大，这样浪费时间以及空间
    由于K一般比m即训练样本的数目少很多，所以这里用向量相乘计算m个样本，外层用K个for循环，即时间复杂度为O（K），而不是O（m）
    '''
    for i in range(K):  #dataY[:,0] 为类别1
        Ak = -1 * dot(dataY[:, i].T, log(hx[:, i]))
        Bk = -1 * dot((1 - dataY[:, i]).T, log(1 - hx[:, i]))
        jtheta += Ak + Bk
    jtheta = jtheta * 1.0 / m
    #print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",jtheta
    reguTerm = getRegularTerm(theta1, theta2, lamda, m)
    jtheta_Regular = jtheta + reguTerm
    print "#################jtheta_Regular", jtheta_Regular
    return jtheta_Regular
    #return jtheta    # 先暂时不用加regular项


def sigmodGradient(z):
    gz = 1.0 / (1 + exp(-z))
    p = 1e-5
    m, n = shape(gz)
    gz_new = []
    for f in gz.flat:
        if f == 1:
            gz_new.append([f - p])
        elif f == 0:
            gz_new.append([f + p])
        else:
            gz_new.append([f])
    gz = asarray(gz_new)
    gz = reshape(gz, (m, n))
    return gz * (1 - gz)


def randInitialTheta(L_out, L_in, epsilon_init):
    theta = random.rand(L_out, L_in) * 2 * epsilon_init - epsilon_init
    return theta


def getActivations(dataX, theta1, theta2):
    # dataX=reshape(dataX,(shape(dataX)[0],1))
    z2 = dot(theta1, dataX)  #25*1
    a2 = sigmodFunction(z2)
    a2_new = concatenate((ones((1, 1)), a2))
    z3 = dot(theta2, a2_new)
    a3 = sigmodFunction(z3)
    # print "^^^^^^^^^^^^^^^^^",z2,a2,a2_new,z3
    # print "%%%%%%%%%%%%%%",a3
    return z2, a2, a2_new, z3, a3


def backPropagation(dataY, z2, a2, a2_new, z3, a3, dataX_Column, theta1, theta2):
    deta3 = a3 - reshape(dataY, (shape(dataY)[0], 1))  # 10*1
    deta2 = dot(theta2.T, deta3) * concatenate((ones((1, 1)), sigmodGradient(z2)))
    # print "#######################",shape(deta2)
    deta2 = deta2[1:, :]  # 去掉deta[0,:] 去掉第一个元素 为bias    此处维度为25*1
    gra1 = dot(deta2, dataX_Column.T)
    gra2 = dot(deta3, a2_new.T)
    return gra1, gra2


def forwardPropagation(theta, *args):  # 包含正则化项
    global num
    input_layer_size = args[0]  # 这里没有加偏置项
    hidden_layer_size = args[1]  # 这里没有加偏置项
    dataX_new = args[2]
    dataY_new = args[3]
    K = args[4]
    lamda = args[5]
    m, n = shape(dataX_new)
    m1 = hidden_layer_size
    n1 = input_layer_size + 1
    m2 = K
    n2 = hidden_layer_size + 1
    theta1 = reshape(theta[0:m1 * n1], (m1, n1))
    theta2 = reshape(theta[m1 * n1:m1 * n1 + m2 * n2], (m2, n2))
    gra_theta1 = 0
    gra_theta2 = 0

    for i in range(m):
        dataX_Column = reshape(dataX_new[i, :], (shape(dataX_new[i, :])[0], 1))
        z2, a2, a2_new, z3, a3 = getActivations(dataX_Column, theta1,
                                                theta2)  # z2 z3 a2 a3没有bias   a2_new dataX_new有bias
        gra1, gra2 = backPropagation(dataY_new[i, :], z2, a2, a2_new, z3, a3, dataX_Column, theta1, theta2)
        gra_theta1 += gra1
        gra_theta2 += gra2
    gra_theta1 = 1.0 * gra_theta1 / m + (lamda * 1.0 / m) * theta1
    gra_theta2 = 1.0 * gra_theta2 / m + (lamda * 1.0 / m) * theta2
    # print  "$$$$$$$$$$$$$$$$",gra_theta1
    #print "$$$$$$$$$$$$$$$$",(lamda*1.0/m)*theta1
    gra_theta1[:, 0] = gra_theta1[:, 0] - (lamda * 1.0 / m) * theta1[:, 0]  # 每一行的第一列为bias项，不需要正则化
    gra_theta2[:, 0] = gra_theta2[:, 0] - (lamda * 1.0 / m) * theta2[:, 0]
    #print  "$$$$$$$$$$$$$$$$",gra_theta1
    #print "$$$$$$$$$$$$$$$$",(lamda*1.0/m)*theta1
    print shape(gra_theta1), shape(gra_theta2)
    num = num + 1
    print "iteration", num
    graFinal = concatenate((gra_theta1.flatten(), gra_theta2.flatten()))
    print "$%$%$%$%$%$%$%$%$", graFinal
    return graFinal


def getArgs(theta, args):
    input_layer_size = args[0]  # 这里没有加偏置项
    hidden_layer_size = args[1]  # 这里没有加偏置项
    dataX_new = args[2]
    dataY_new = args[3]
    K = args[4]
    lamda = args[5]
    m, n = shape(dataX_new)
    m1 = hidden_layer_size
    n1 = input_layer_size + 1
    m2 = K
    n2 = hidden_layer_size + 1
    theta1 = reshape(theta[0:m1 * n1], (m1, n1))
    theta2 = reshape(theta[m1 * n1:m1 * n1 + m2 * n2], (m2, n2))
    return theta1, theta2, dataX_new, K, dataY_new, m


def computeNumericalGradient(theta, *args):
    numgrad = zeros(shape(theta))
    perturb = zeros(shape(theta))
    # print "###########################",perturb,shape(perturb)
    e = 1e-4  # gradient checking 中设置的参数
    for i in range(shape(theta)[0]):
        perturb[i] = e
        loss1 = costFunction_Regular(theta - perturb, *args)
        loss2 = costFunction_Regular(theta + perturb, *args)
        numgrad[i] = (loss2 - loss1) * 1.0 / (2 * e)
        print "#$#$#$#$#$#$#", loss1
        print "#$#$#$#$#$#$#", loss2
        print "#$#$#$#$#$#$#", numgrad[i]
        perturb[i] = 0
    return numgrad


def debugInitialWeights(hidden_layer_size, input_layer_size):
    W = zeros((hidden_layer_size, input_layer_size + 1))
    W = reshape(sin(range(1, shape(W.flatten())[0] + 1)), shape(
        W)) / 10  # sin范围[-1,1]  初始10 让theta以及初始数据的范围都为[-0.1,0.1]。平时正常计算的时候theta的取值范围为[-0.12,0.12], 输入数据feature scaling之后取值范围为[-0.5,0.5]
    # print "@@@@@@@@@@@@@@@@@",W
    return W


def processGradientCheck():
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    theta1 = debugInitialWeights(hidden_layer_size, input_layer_size)
    theta2 = debugInitialWeights(num_labels, hidden_layer_size)
    X = debugInitialWeights(m, input_layer_size - 1)
    dataX_new = concatenate((ones((m, 1)), X), axis=1)
    y = 1 + remainder(range(1, m + 1), num_labels)
    dataY_new = preProcessDataY(y, m, num_labels)
    lamda = 1
    theta = concatenate((theta1.flatten(), theta2.flatten()))
    numgrad = computeNumericalGradient(theta,
                                       *(input_layer_size, hidden_layer_size, dataX_new, dataY_new, num_labels, lamda))
    print "#######################numgrad###############", numgrad, shape(numgrad)
    graFinal = forwardPropagation(theta,
                                  *(input_layer_size, hidden_layer_size, dataX_new, dataY_new, num_labels, lamda))
    print "###################graFinal###############", graFinal, shape(graFinal)
    diff = LA.norm(numgrad - graFinal) * 1.0 / LA.norm(numgrad + graFinal)
    print "#####################diff##################", diff


def saveModelFiles(res):
    f = open("data/nn_models_iter100", "w")
    res_bin = pickle.dumps(res, 1)
    pickle.dump(res_bin, f)
    f.close()


def getModelFiles():
    f = open("data/nn_models_iter100", "r")
    res_bin = pickle.load(f)
    res = pickle.loads(res_bin)
    f.close()
    return res


if __name__ == "__main__":
    print "begin neural network "
    path = "data/nn/ex4_handwrite_digit.mat"
    K = 10  # 10个类别
    lamda = 1
    input_layer_size = 400
    hidden_layer_size = 25
    dataX, dataY = loadDataSet(path)
    print  "handwrite", dataX
    print  "handwrite", shape(dataX)

    path = "data/nn/ex4_nnweights.mat"
    theta1, theta2 = loadWeights(path)
    # print "提供的theta1",theta1
    #print "提供的theta2",theta2
    theta = concatenate((theta1.flatten(), theta2.flatten()))
    m, n = shape(dataX)
    dataX_new = concatenate((ones((m, 1)), dataX), axis=1)
    dataY_new = preProcessDataY(dataY, m, K)  # 将每一个训练样本的类别数变成[0,0,0,0,...1,0]向量的形式    5000*10
    showDataFigure(random.permutation(dataX)[0:100, :])  # 随机取出100条数据
    jtheta = costFunction_Regular(theta, *(input_layer_size, hidden_layer_size, dataX_new, dataY_new, K, lamda))
    print "初始theta得到的jtheta为:", jtheta
    #print "sigmodGradient:",sigmodGradient(0)
    epsilon_init = 0.12  #一个很小的接近0的数
    theta1 = randInitialTheta(hidden_layer_size, input_layer_size + 1, epsilon_init)
    theta2 = randInitialTheta(K, hidden_layer_size + 1, epsilon_init)
    theta = concatenate((theta1.flatten(), theta2.flatten()))
    ##forwardPropagation(theta,*(input_layer_size,hidden_layer_size,dataX_new,dataY_new,K,lamda))
    print "开始执行梯度检查"
    processGradientCheck()
    res = fmin_cg(costFunction_Regular, theta, forwardPropagation,
                  (input_layer_size, hidden_layer_size, dataX_new, dataY_new, K, lamda), maxiter=100, full_output=1,
                  retall=1)
    saveModelFiles(res)
    res = getModelFiles()
    print "res", res[0]
    print res[1]
    print shape(res[0]), type(res[0])
    print (res[0])[4:10]
    preClass = predictAccurancy(dataX_new, res[0], input_layer_size, hidden_layer_size, K)
    acc = sum(preClass == dataY.flat) * 1.0 / m
    print "准确度为:", acc
    theta1_model = reshape(res[0][0:(input_layer_size + 1) * hidden_layer_size],
                           (hidden_layer_size, input_layer_size + 1))
    theta1_model_nobias = theta1_model[:, 1:]
    showDataFigure(theta1_model_nobias)
    ###########################展示##############################
    dataX_Random = random.permutation(dataX)
    dataX_Random_New = concatenate((ones((m, 1)), dataX_Random), axis=1)
    for i in arange(0, m):
        print "预测的结果为:"
        print predictAccurancy(reshape(dataX_Random_New[i, :], (1, 401)), res[0], input_layer_size, hidden_layer_size,
                               K)
        showDataFigure(reshape(dataX_Random[i, :], (1, 400)))