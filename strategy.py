# import importlib
# importlib.reload(utils)
import utils
import numpy as np

def MAStrategy(data:"pandas frame",days:int):
    """"
    收盘价连续两天在均线上则买入，连续两天在均线下则卖出
    flag主要用于回测，所以flag应该是前面的数据计算的结果，
    今天的操作应该根据flag来进行，flag=1，则开盘买入，flag=-1，开盘卖出
    """
    flagList = np.zeros(len(data['open']))
    ma = utils.MA(data,days)
    for i in range(2,len(flagList)):
        if data['close'][i-1]>ma[i-1] and data['close'][i-2]>ma[i-2]:
            flagList[i] = 1
        elif data['close'][i-1]<ma[i-1] and data['close'][i-2]<ma[i-2]:
            flagList[i] = -1
    return flagList


# 连续三天收盘价增高或者降低，就添加趋势翻转标记
def myMomentDot(data):
    flagList = np.zeros(len(data['open']))
    for i in range(4, len(data["Close"])):
        moment = 0
        for j in range(3):
            if data["Close"][i - j - 1] > data["Close"][i - j - 2]:
                moment += 1
        if moment == 3:
            flagList[i] = 1
        elif moment == 0:
            flagList[i] = -1

    return flagList

# myMomentDot策略添加一个条件：当天下跌超过3%的卖出
def myMomentDot2(data):
    flagList = np.zeros(len(data['open']))
    for i in range(4, len(data["Close"])):
        moment = 0
        for j in range(3):
            if data["Close"][i - j - 1] > data["Close"][i - j - 2]:
                moment += 1
        if moment == 3 or (float(data["Open"][i - 1]) - float(data["Close"][i - 1])) / float(data["Open"][i - 1]) > 0.03:
            flagList[i] = 1
        elif moment == 0:
            flagList[i] = -1

    return flagList



