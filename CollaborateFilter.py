__author__ = 'wangxiang'
# -*- coding:utf8 -*-
import scipy.io as sio
from numpy import shape, mean, dot, sum, asarray, concatenate, reshape, zeros, linalg as LA, nonzero, random, \
    count_nonzero, \
    argsort
from scipy.optimize import fmin_cg, fmin_bfgs
import datetime


class CollaborateFilter(object):
    def __init__(self, path):
        self.data = sio.loadmat(path)
        # print self.data
        self.Y = self.data['Y']
        self.R = self.data['R']
        # print self.R[0,:]==1
        # print mean(self.Y[0,self.R[0,:]==1]) #求Y中第一行中评过分数的分数平均值
        # print mean(self.Y[1681,self.R[1681,:]==1])
        # print mean(self.Y[0,self.Y[0,:]!=0])
        # print mean(self.Y[1,self.R[1,:]==1])

    def loadMoiveParams(self, path):
        data = sio.loadmat(path)
        self.num_features = data["num_features"]
        # print "num_features :",type(self.num_features),shape(self.num_features),self.num_features
        self.num_users = data["num_users"]
        # print "num_users :",shape(self.num_users),self.num_users
        self.X = data["X"]
        # print "self.X 的维度大小",shape(self.X)
        self.Theta = data["Theta"]
        # print "self.Theta 的维度大小",shape(self.Theta)
        self.num_movies = data["num_movies"]
        print shape(self.num_movies), self.num_movies

    def testPartData(self):
        self.num_movies = 5
        self.num_features = 3
        self.num_users = 4
        self.X = self.X[0:self.num_movies, 0:self.num_features]
        self.Theta = self.Theta[0:self.num_users, 0:self.num_features]
        self.Y = self.Y[0:self.num_movies, 0:self.num_users]
        self.R = self.R[0:self.num_movies, 0:self.num_users]

    def getCostFunction(self, lamda):
        res = dot(self.X, self.Theta.T)
        res = (res - self.Y) * (res - self.Y)
        self.Jtheta = sum(res[self.R[:] == 1]) / 2.0
        regX_part = dot(self.X.flatten(), self.X.flatten().T)
        regTheta_part = dot(self.Theta.flatten(), self.Theta.flatten().T)  # 参数Theta的部分正则化项
        reg = lamda / 2.0 * (regX_part + regTheta_part)
        self.Jtheta = self.Jtheta + reg
        print "self.Jtheta$$$$$$$$$$$$$$$$$", self.Jtheta
        # res2=0
        # for i in range(0,self.num_movies):
        # for j in range(0,self.num_users):
        # if(self.R[i,j]==1):
        #             res2+=pow(dot(self.Theta[j,:],self.X[i,:].T)-self.Y[i,j],2)
        # res2=res2/2.0
        # print "@##@#@#@##@#@#@##@#",res2
        # res=dot(self.X,self.Theta.T)
        # res=(res*self.R-self.Y)*(res*self.R-self.Y)
        # print "@#@##@##@#@#@#@#@#@####################################",sum(res)/2.0

    def getGradient(self, lamda):
        res = dot(self.X, self.Theta.T)
        res[self.R[:] == 0] = 0
        gra_Jx = dot((res - self.Y), self.Theta)
        reg_x_part = lamda * self.X
        self.gra_Jx = gra_Jx + reg_x_part
        gra_Jtheta = dot((res - self.Y).T, self.X)
        reg_theta_part = lamda * self.Theta
        self.gra_Jtheta = gra_Jtheta + reg_theta_part  # #cost function 对所有人（self.theta）的梯度 nu*3   包含正则化项

    def checkGradient(self, lamda):
        res = concatenate((self.X, self.Theta)).flatten()  # 将两个ndarray连接起来，默认axis=0，是按照列进行连接
        deta = 1e-4
        numgrad = []
        preturb = zeros(shape(res))
        for i in range(0, 27):
            preturb[i] = deta
            # resPlus=res
            # resPlus[i]=resPlus[i]+deta
            # print "resPlus[i]######",resPlus[i]
            # resMinus=res
            # print "^^^^^^^^^^^^^^^^^resMinus",resMinus
            # resMinus[i]=resMinus[i]-deta
            # print "resMinus[i]######",resMinus[i]
            temp_result = res + preturb
            # print "plus  temp_result $$$$",temp_result
            x = reshape(temp_result[0:15], (5, 3))
            theta = reshape(temp_result[15:27], (4, 3))
            tem = dot(x, theta.T)
            tem = (tem - self.Y) * (tem - self.Y)
            Jtheta_Plus = sum(tem[self.R[:] == 1]) / 2.0

            regX_part = dot(x.flatten(), x.flatten().T)
            regTheta_part = dot(theta.flatten(), theta.flatten().T)
            reg = lamda / 2.0 * (regX_part + regTheta_part)
            Jtheta_Plus = Jtheta_Plus + reg
            # print "Jtheta_Plus######",Jtheta_Plus
            temp_result = res - preturb
            # print "minus  temp_result $$$$",temp_result
            x = reshape(temp_result[0:15], (5, 3))
            theta = reshape(temp_result[15:27], (4, 3))
            tem = dot(x, theta.T)
            tem = (tem - self.Y) * (tem - self.Y)
            Jtheta_Minus = sum(tem[self.R[:] == 1]) / 2.0
            regX_part = dot(x.flatten(), x.flatten().T)
            regTheta_part = dot(theta.flatten(), theta.flatten().T)
            reg = lamda / 2.0 * (regX_part + regTheta_part)
            Jtheta_Minus = Jtheta_Minus + reg
            # print "Jtheta_Minus######",Jtheta_Minus
            numgrad.append((Jtheta_Plus - Jtheta_Minus) * 1.0 / (2 * deta))
            preturb[i] = 0
        numgrad = asarray(numgrad)  # 得到的近似梯度
        gra = concatenate((self.gra_Jx, self.gra_Jtheta)).flatten()  # 通过计算偏导得到的梯度，即我们要检查的梯度
        print "numgrad is ######", numgrad
        print "gra is #####", gra
        print LA.norm(numgrad - gra)
        print LA.norm(numgrad + gra)
        print "范数比较", LA.norm(numgrad - gra) * 1.0 / LA.norm(numgrad + gra)

    def loadMoiveList(self):
        fmoives = open('data/cf/ex8_movie_ids.txt')
        moives = []
        for m in fmoives.readlines():
            try:
                moives.append(m[m.find(" ") + 1:].strip("\n"))
            except Exception, e:
                print "exception", e
        fmoives.close()
        self.moives = reshape(asarray(moives), (len(moives), 1))

    def setOwnRating(self):
        self.my_ratings = zeros(shape(self.moives))
        self.my_ratings[0, 0] = 4
        self.my_ratings[97, 0] = 2
        self.my_ratings[6, 0] = 3
        self.my_ratings[11, 0] = 5
        self.my_ratings[53, 0] = 4
        self.my_ratings[63, 0] = 5
        self.my_ratings[65, 0] = 3
        self.my_ratings[68, 0] = 5
        self.my_ratings[182, 0] = 4
        self.my_ratings[225, 0] = 5
        self.my_ratings[354, 0] = 5
        for i in range(0, shape(self.my_ratings)[0]):
            if self.my_ratings[i, 0] > 0:
                print 'Rated %.2f for %s' % (self.my_ratings[i, 0], self.moives[i, 0])

    def loadMoives(self, path):
        self.data = sio.loadmat(path)
        self.Y = self.data['Y']
        self.R = self.data['R']

    def meanNormalization(self):
        m, n = shape(self.Y)
        print "##################", m, n
        meanY = zeros((m, 1))
        print "###初始meanY", meanY
        normY = zeros((m, n))
        '''
        for i in range (shape(self.Y)[0]):#
            meanY[i,0]=mean(self.Y[i,self.R[i,:]==1])
            idx=[k for k in range(n) if self.R[i,k]!=0]  # 获取每个人对电影i评过分的索引
            normY[i,idx]=self.Y[i,idx]-meanY[i,0]
        '''
        for i in range(shape(self.Y)[0]):  #
            meanY[i, 0] = mean(self.Y[i, self.R[i, :] == 1])
            idx = nonzero(self.R[i, :])[0]  # 由于是for循环遍历，每次nonzero返回的tuple中只有一个元素,idx为一个数组
            normY[i, idx] = self.Y[i, idx] - meanY[i, 0]

            # for i in range (shape(self.Y)[0]):#
            # temp=mean(self.Y[i,self.R[i,:]==1])
            # meanY.append(temp)
            # normY.append(self.Y[i,self.Y[i,:]!=0]-temp)

        self.meanY = meanY
        self.normY = normY
        # print shape(self.meanY)
        # self.normY=self.Y[self.R==1]-self.meanY
        # print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",self.Y
        # print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",self.meanY
        # print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",self.normY

    def randInitialParam(self, featureNum):
        moives, users = shape(self.Y)
        self.Theta = random.randn(users, featureNum)  # randn 生成users行featureNum列的符合标准正太分布的矩阵
        self.X = random.randn(moives,
                              featureNum)  # 之前写的神经网络随机初始话参数的方法是 制定一个deta很小例如0.12， rand随机初始化一个[0,1)的数据，即 2rand(a,b)deta-deta
        return concatenate((self.Theta.flatten(), self.X.flatten()))

    '''
    下面两个方法costFunctionc_Regular,gradient_Regular都是在上面getCostFunction 以及getGradient的基础上小改一下
    '''

    def costFunction_Regular(self, initParams, *args):
        lamda = args[0]
        moives = args[1]
        users = args[2]
        features = args[3]

        Theta = reshape(initParams[0:users * features], (users, features))  # initParams Theta 参数在前 X 在后
        X = reshape(initParams[users * features:], (moives, features))
        # print "###########costFunctionc_Regular 初始化参数Theta为@@@@@@@@!!!!!",Theta ,shape(Theta)
        # print "###########costFunctionc_Regular 初始化参数X为@@@@@@@@!!!!!",X,shape(X)
        res = dot(X, Theta.T)
        res = (res - self.normY) * (res - self.normY)
        Jtheta = sum(res[self.R[:] == 1]) / 2.0
        # print "costFunctionc_Regular 中Jtheta的值为:",Jtheta
        regX_part = sum(X * X)
        # print "costFunctionc_Regular 中regX_part的值为:",regX_part
        # regTheta_part=dot(Theta.flatten(),Theta.flatten().T)
        regTheta_part = sum(Theta * Theta)
        # print "costFunctionc_Regular 中regTheta_part的值为:",regTheta_part
        reg = lamda / 2.0 * (regX_part + regTheta_part)

        Jtheta = Jtheta + reg
        # print "Jtheta$$$$$$$$$$$$$$$$$",Jtheta,type(Jtheta),shape(Jtheta)
        return Jtheta

    def gradient_Regular(self, initParams, *args):
        lamda = args[0]
        moives = args[1]
        users = args[2]
        features = args[3]
        Theta = reshape(initParams[0:users * features], (users, features))  # initParams Theta 参数在前 X 在后
        X = reshape(initParams[users * features:], (moives, features))
        res = dot(X, Theta.T)
        res[self.R[:] == 0] = 0
        gra_Jx = dot((res - self.normY), Theta)
        reg_x_part = lamda * X
        gra_Jx = gra_Jx + reg_x_part
        gra_Jtheta = dot((res - self.normY).T, X)
        reg_theta_part = lamda * Theta
        gra_Jtheta = gra_Jtheta + reg_theta_part  # #cost function 对所有人（self.theta）的梯度 nu*3
        # print "gra_Jtheta:",gra_Jtheta
        # print "gra_Jx",gra_Jx
        gra = concatenate((gra_Jtheta.flatten(), gra_Jx.flatten()))
        # print "gra",type(gra),gra
        return gra

    def getPredictResults(self):
        self.PredictY = dot(self.X, self.Theta.T) + self.meanY
        # print "self.meanY",self.meanY
        # print "self.Y",count_nonzero(self.R[:,943])
        # print "self.normY",self.normY
        # print "getPredictResults的self.PredictY为",self.PredictY
        # print "getPredictResults的self.PredictY为",self.PredictY+self.meanY
        # 去除myown的预测评分
        myOwnPredict = self.PredictY[:, shape(self.normY)[1] - 1]
        print "myOwnPredict ", myOwnPredict
        # 挑选出以前没有评过分数的
        myOwnNoRate = self.PredictY[self.R[:, shape(self.normY)[1] - 1] == 0, shape(self.normY)[1] - 1]
        #对评分进行降序排序
        resIdx = argsort(-myOwnPredict)  # 返回降序排序的索引
        resNoRateIdx = argsort(-myOwnNoRate)
        print "排完顺序之后的结果为:", myOwnPredict[resIdx]
        print "@@@@@@@@@@@@@@@@@@@@", resIdx, type(resIdx)
        print "排完顺序之后的结果为(以前没有评分):", myOwnNoRate[resNoRateIdx]
        print "@@@@@@@@@@@@@@@@@@@@", resNoRateIdx, type(resNoRateIdx)
        #resTopTen=resIdx[0:30]
        resTopTen = resNoRateIdx[0:30]  #从以前没有没有评分的电影中挑选中前30
        for i in range(30):
            print 'Predicting rating %.1f for movie %s\n' % (myOwnNoRate[resTopTen[i]], self.moives[resTopTen[i], 0])
        print  myOwnNoRate[resTopTen]
        print  self.moives[resTopTen, 0]
        for i in range(0, shape(self.my_ratings)[0]):
            if self.my_ratings[i, 0] > 0:
                print 'Rated %.2f for %s' % (self.my_ratings[i, 0], self.moives[i, 0])


if __name__ == "__main__":
    # 创建CollaborateFilter对象，初始化数据参数
    path = "data/cf/ex8_movies.mat"
    cf = CollaborateFilter(path)  # 读取 Y R
    # 加载已经存在的参数
    path = "data/cf/ex8_movieParams.mat"
    cf.loadMoiveParams(path)  # 1682*943 10个特征
    # 为了让模型运行的跟快一些，这里手动将各个参数调小
    cf.testPartData()
    lamda = 0
    lamda = 1.5
    cf.getCostFunction(lamda)
    cf.getGradient(lamda)
    print " cf.gra_Jx:\n"
    print cf.gra_Jx
    print "cf.gra_Jtheta:\n"
    print cf.gra_Jtheta
    #梯度检查，保证我们计算的梯度是正确的，目的是保证不会对预测结果产生不好的影响
    # cf.checkGradient(lamda)
    '''
     上面主要是计算cost function以及梯度，使用的是一部分的数据，这样方便gradient checking
     下面开始推荐电影 （注释掉cf.testPartData()）  costFunction 以及计算梯度的方法getGradient 都重新写，因为上面的做一些测试以及实验，下面正式计算的时候稍微改一下
    '''
    print "################################################################################################################################################"
    # 加载电影列表
    cf.loadMoiveList()
    cf.setOwnRating()
    #重新加载电影数据（按照教程循序渐进）
    cf.loadMoives("data/cf/ex8_movies.mat")
    cf.Y = concatenate((cf.Y, cf.my_ratings), axis=1)  # 将自己的评分加入到self.Y中
    print "cf.y", cf.Y, shape(cf.Y)
    cf.R = concatenate((cf.R, cf.my_ratings != 0), axis=1)  # 将自己对应的R加入到self.R中
    print cf.R, shape(cf.R)
    #计算self.Y每一行的平均值，然后meannormalization  解决冷启动的问题
    cf.meanNormalization()
    features = 10  # 这里自己定义的10个特征
    #随机初始化参数，不能全部默认为0，如果为零则每次所有参数都是零，没有意义
    initParams = cf.randInitialParam(features)

    print "initParmas###########", initParams, type(initParams), shape(initParams)
    lamda = 10
    m, n = shape(cf.Y)
    beginTime = datetime.datetime.now()
    print "当前时间为:", beginTime.strftime("%Y%m%d %H:%M:%S")
    xopt, fopt, func_calls, grad_calls, warnflag, allvecs = fmin_cg(cf.costFunction_Regular, initParams,
                                                                    cf.gradient_Regular, (lamda, m, n, features),
                                                                    maxiter=100, full_output=1, retall=1)
    endTime = datetime.datetime.now()
    print "当前时间为:", endTime.strftime("%Y%m%d %H:%M:%S")
    print "fmin_cg消耗的时间为 :", (endTime - beginTime)
    print xopt, fopt, func_calls, grad_calls, warnflag
    cf.Theta = reshape(xopt[0:n * features], (n, features))
    cf.X = reshape(xopt[n * features:], (m, features))
    cf.getPredictResults()





