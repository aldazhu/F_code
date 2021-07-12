import numpy as np

def testDaily(flagList,data):
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
    currentEarningRatio = []

    for i in range(len(flagList)):
        if stockPool==0 and flagList[i]==1:
            stockPool = 1
            buyPrice = data['open'][i]
            buyNum += 1
        elif stockPool==1 and flagList[i]==-1:
            sellPrice = data['open'][i]
            stockPool = 0
            changePercent += (sellPrice-buyPrice)/buyPrice - tradeFeeRatio
            changePrice += sellPrice - buyPrice(1+tradeFeeRatio)
            sellNum += 1
        if stockPool == 1:
            currentEarningRatio[i] = (data['close'][i] - buyPrice)/buyPrice
        elif stockPool == 0:
            #如果没有买股票则收益率保持和前面一天一致，第一天为0
            if i==0:
                currentEarningRatio[i] = 0
            else:
                currentEarningRatio[i] = currentEarningRatio[i-1]
    return currentEarningRatio


