import numpy as np

def test(flagList,data):
    """"
    这里的测试逻辑，在看历史数据后对第二天的状态进行预测，
    如果flag=1 且stockPool=0
    则以开盘价买入，否则以开盘价卖出。
    部分开盘就10%涨停的是买不了的
    """
    stockPool = 0
    tradeFeeRatio = 0.001
    changePercent = 0
    changePrice = 0
    buyNum = 0
    sellNum = 0
    buyPrice = 0
    sellPrice = 0
    dayRatio = np.zeros(len(flagList))

    for i in range(1, len(flagList)-1):
        #buy
        if stockPool == 0 and flagList[i] == 1:
            stockPool = 1
            buyPrice = data['open'][i]
            buyNum += 1
            dayRatio[i] = (data['close'][i] - data['open'][i])/data['open'][i]
        #sell
        if stockPool == 1 and flagList[i] == -1:
            sellPrice = data['open'][i]
            stockPool = 0
            sellNum += 1
            changePercent += (sellPrice - buyPrice) / buyPrice - tradeFeeRatio
            dayRatio[i] = (data['open'][i] - data['close'][i-1])/data['close'][i-1] - tradeFeeRatio
        #hold
        if stockPool == 1 and flagList[i] != -1:
            dayRatio[i] = (data['close'][i] - data['close'][i-1])/data['close'][i-1]
        #empty
        if stockPool == 0 and flagList[i] != 1:
            dayRatio[i] = 0
    accRatio = np.zeros(len(dayRatio))
    for i in range(1,len(dayRatio)):
        accRatio[i] = accRatio[i-1] + dayRatio[i]

    return accRatio,changePercent

def testPool(data_list:"list of pandas frame ", flag_List):
    """
    批量测试
    """

