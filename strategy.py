# import importlib
# importlib.reload(utils)
import utils
import numpy as np
import indicator

def upMATwoDays(data:"pandas frame",days:int):
    """"
    收盘价连续两天在均线上则买入，连续两天在均线下则卖出
    flag主要用于回测，所以flag应该是前面的数据计算的结果，
    今天的操作应该根据flag来进行，flag=1，则开盘买入，flag=-1，开盘卖出
    """
    flagList = np.zeros(len(data['open']))
    ma = indicator.MA(data,days)
    for i in range(2,len(flagList)):
        if data['close'][i-1]>ma[i-1] and data['close'][i-2]>ma[i-2]:
            flagList[i] = 1
        elif (data['close'][i-1]<ma[i-1] and data['close'][i-2]<ma[i-2]) or (data['close'][i-1]-data['open'][i-1])/data['open'][i-1] <= -0.03:
            flagList[i] = -1
    return flagList

def MATrend(data:"pandas frame",days:int):
    """"
    收盘在ma上连续两天，且ma连续两天向上则买入
    收盘在ma下连续两天，或者ma向下连续两天则卖出
    flag主要用于回测，所以flag应该是前面的数据计算的结果，
    今天的操作应该根据flag来进行，flag=1，则开盘买入，flag=-1，开盘卖出
    """
    thresh = -0.03#控制当天下跌
    flagList = np.zeros(len(data['open']))
    ma = indicator.MA(data,days)
    for i in range(2,len(flagList)):
        #是否连续两天在ma上
        upMATwoDays_flag = data['close'][i-1]>ma[i-1] and data['close'][i-2]>ma[i-2]
        #ma是否向上运行
        MATrend_up = ma[i-1]>ma[i-2]
        #是否连续两天在ma下
        downMATwoDays_flag = (data['close'][i - 1] < ma[i - 1] and data['close'][i - 2] < ma[i - 2])
        #ma是否向下运行
        MATrend_down = ma[i-1]<ma[i-2]
        #当日下跌是否超过阈值
        chg_flg  = (data['close'][i - 1] - data['open'][i - 1]) / data['open'][i - 1] <= thresh
        if  upMATwoDays_flag and MATrend_up :
            flagList[i] = 1
        elif  downMATwoDays_flag or MATrend_down :
            flagList[i] = -1
    return flagList

def upMA(data:"pandas frame",days:int):
    """"
    收盘在ma上连续两天，buy
    收盘在ma下连续两天，sell
    flag主要用于回测，所以flag应该是前面的数据计算的结果，
    今天的操作应该根据flag来进行，flag=1，则开盘买入，flag=-1，开盘卖出
    """
    thresh = -0.03#控制当天下跌
    flagList = np.zeros(len(data['open']))
    ma = indicator.MA(data,days)
    for i in range(2,len(flagList)):
        #是否连续两天在ma上
        upMATwoDays_flag = data['close'][i-1]>ma[i-1] and data['close'][i-2]>ma[i-2]

        #是否连续两天在ma下
        downMATwoDays_flag = (data['close'][i - 1] < ma[i - 1] and data['close'][i - 2] < ma[i - 2])

        #当日下跌是否超过阈值
        chg_flg  = (data['close'][i - 1] - data['open'][i - 1]) / data['open'][i - 1] <= thresh
        if  upMATwoDays_flag  :
            flagList[i] = 1
        elif  downMATwoDays_flag  :
            flagList[i] = -1
    return flagList


# 连续三天收盘价增高或者降低，就添加趋势翻转标记
def myMomentDot(data):
    flagList = np.zeros(len(data['open']))
    for i in range(4, len(data["close"])):
        moment = 0
        for j in range(3):
            if data["close"][i - j - 1] > data["close"][i - j - 2]:
                moment += 1
        if moment == 3:
            flagList[i] = 1
        elif moment == 0:
            flagList[i] = -1
    return flagList

# myMomentDot策略添加一个条件：当天下跌超过3%的卖出
def myMomentDot2(data):
    flagList = np.zeros(len(data['open']))
    for i in range(4, len(data["close"])):
        moment = 0
        for j in range(3):
            if data["close"][i - j - 1] > data["close"][i - j - 2]:
                moment += 1
        if moment == 3 :
            flagList[i] = 1
        elif moment == 0 or (float(data["open"][i - 1]) - float(data["close"][i - 1])) / float(data["open"][i - 1]) > 0.03:
            flagList[i] = -1

    return flagList


def delmyMomentDot2(d):
    #错误的，提前知道未来三天一定是连续增长或连续下降的，回归结果非常好
    date = []
    close_p = []
    open_p = []  # 今日开盘价
    flg_l = np.zeros(len(d['open']))
    flg = 0
    for i in range(4, len(d["close"])):
        moment = 0
        for j in range(3):
            if d["close"][i - j - 1] > d["close"][i - j - 2]:
                moment += 1
        if moment == 3:
            flg = 1
            flg_l[i-3]= flg
            close_p= (d["close"][i])
            open_p= (d["open"][i])
        elif moment == 0 or (float(d["open"][i - 1]) - float(d["close"][i - 1])) / float(d["open"][i - 1]) > 0.03:
            flg = -1
            flg_l[i-3] = flg
            close_p = (d["close"][i])
            open_p = (d["open"][i])

        else:
            flg = 0
            flg_l[i-3] = flg
            close_p = (d["close"][i])
            open_p = (d["open"][i])

    return flg_l

def doubleMA(data,fastMADays=5,slowMADays=15):
    fastMA = indicator.MA(data,fastMADays)
    slowMA = indicator.MA(data,slowMADays)

    flagList = np.zeros(len(data['open']))

    for i in range(slowMADays,len(data['open'])):
        if fastMA[i-2]<slowMA[i-2] and fastMA[i-1]>slowMA[i-1]:
            flagList[i] = 1
        elif fastMA[i-2]>slowMA[i-2] and fastMA[i-1]<slowMA[i-1]:
            flagList[i] = -1

    return flagList

def upupgo(data):
    '''

    '''
    flagList = np.zeros(len(data['open']))
    for i in range(5, len(data['open'])):
        flg=0
        for j in range(3):
            if data['open'][i-j-1] > data['close'][i-j-1]:
                flg+=1
        if flg==3:
            flagList[i] = 1
        elif flg==0:
            flagList[i] = -1
    return flagList


def CCIThresh(data, days=5,CCI_thresh = 100, after_Ndays=10):
    """
    当CCI<-100 buy,after N days sell, CCI > 100 sell
    """
    CCI = indicator.CCI(data,days)
    flagList_buy = np.where(CCI < -CCI_thresh, 1, 0)
    flagList_sell = 0 - np.where(CCI > CCI_thresh,-1, 0)
    flagList = flagList_buy + flagList_sell
    for i in range(days,len(flagList)-after_Ndays):
        if flagList[i] == 1:
            flagList[i+after_Ndays] = -1
    return flagList
