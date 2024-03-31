import numpy as np

def test(flagList,data):
    """"
    这里的测试逻辑，在看历史数据后对第二天的状态进行预测，
    如果flag=1 且stockPool=0
    则以开盘价买入，flag=-1以开盘价卖出。
    部分开盘就10%涨停的是买不了的
    """
    stockPool = 0
    tradeFeeRatio = 0.0003
    changePercent = 0
    changePrice = 0
    buyNum = 0
    sellNum = 0
    buyPrice = 0
    sellPrice = 0
    dayRatio = np.zeros(len(flagList))
    hold_days = 0
    stop_loss_ratio = -0.08 #止损
    folow_stop_loss_ratio = -0.08 #跟随止损
    max_earning = -99999999 

    for i in range(1, len(flagList)-1):
        #buy
        if stockPool == 0 and flagList[i] == 1:
            stockPool = 1
            buyPrice = data['open'][i+1]
            buyNum += 1
            dayRatio[i] = (data['open'][i] - buyPrice) / buyPrice - tradeFeeRatio
            print("buyPrice:",buyPrice)
        #sell
        if stockPool == 1 and flagList[i] == -1:
            sellPrice = data['open'][i+1]
            stockPool = 0
            sellNum += 1
            changePercent += (sellPrice - buyPrice) / buyPrice - tradeFeeRatio
            dayRatio[i] = (data['open'][i] - data['close'][i-1])/data['close'][i-1] - tradeFeeRatio
            print("sellPrice:",sellPrice,"changePercent:",(sellPrice - buyPrice) / buyPrice)

        #hold
        if stockPool == 1 and flagList[i] != -1:
            dayRatio[i] = (data['close'][i] - data['close'][i-1])/data['close'][i-1]
            hold_days += 1
            gain = (data['close'][i] - buyPrice) / buyPrice
            
            if  (gain < stop_loss_ratio):
                sellPrice = data['open'][i+1]
                stockPool = 0
                sellNum += 1
                changePercent += (sellPrice - buyPrice) / buyPrice - tradeFeeRatio
                print("sellPrice:",sellPrice,"changePercent:",(sellPrice - buyPrice) / buyPrice)
                print("stop loss")
        #empty
        if stockPool == 0 and flagList[i] != 1:
            dayRatio[i] = 0
    accRatio = np.zeros(len(dayRatio))
    for i in range(1,len(dayRatio)):
        accRatio[i] = accRatio[i-1] + dayRatio[i]

    print("total days",len(flagList),"buyNum:",buyNum,"sellNum:",sellNum,"hold_days:",hold_days)
    return accRatio,changePercent

def testPool(data_list:"list of pandas frame ", flag_List):
    """
    批量测试
    """

