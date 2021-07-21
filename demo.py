import dataPro as dp
import strategy as sg
import test
import utils
import os
import numpy as np

def testMA(file,startDate,endDate,MAdays=5):
    #ma均线策略
    #data = dp.downloadData(stockCode,startDate,endDate)
    data = dp.readData(file)
    data = dp.getNormalData(data)
    flagList = sg.upMATwoDays(data,MAdays)
    earningList,changePercent = test.test(flagList,data)
    print("changePercent:",changePercent)
    #utils.plotEarningRatio(earningList,flagList,data)
    return earningList[-1]

def testMATrend(file,startDate,endDate,MAdays=5):
    #ma均线策略
    #data = dp.downloadData(stockCode,startDate,endDate)
    data = dp.readData(file)
    data = dp.getNormalData(data)
    flagList = sg.MATrend(data,MAdays)
    earningList,changePercent = test.test(flagList,data)
    print("changePercent:",changePercent)
    #utils.plotEarningRatio(earningList,flagList,data)
    return earningList[-1]

def testMoment(file,startDate,endDate):

    #data = dp.downloadData(stockCode,startDate,endDate)
    data = dp.readData(file)
    data = dp.getNormalData(data)
    flagList = sg.myMomentDot2(data)
    earningList,changePercent = test.test(flagList,data)
    print("changePercent:",changePercent)
    #utils.plotEarningRatio(earningList,flagList,data)
    return earningList[-1]

def testDoubleMA(file,startDate,endDate,fastMADays=3,slowMADays=10):

    #data = dp.downloadData(stockCode,startDate,endDate)
    data = dp.readData(file)
    data = dp.getNormalData(data)
    flagList = sg.doubleMA(data,fastMADays,slowMADays)
    earningList,changePercent = test.test(flagList,data)
    print("changePercent:",changePercent)
    #utils.plotEarningRatio(earningList,flagList,data)
    return earningList[-1]

def testUpupGo(file,startDate,endDate):
    #data = dp.downloadData(stockCode,startDate,endDate)
    data = dp.readData(file)
    data = dp.getNormalData(data)
    flagList = sg.upupgo(data)
    earningList,changePercent = test.test(flagList,data)
    print("changePercent:",changePercent)
    #utils.plotEarningRatio(earningList,flagList,data)
    return earningList[-1]

def testGroup(file,startDate,endDate):
    # data = dp.downloadData(stockCode,startDate,endDate)
    data = dp.readData(file)
    data = dp.getNormalData(data)
    flagList_upupgo = sg.upupgo(data)
    flagList_doubleMA = sg.doubleMA(data,7,14)
    flagList_m = sg.myMomentDot(data)

    flagList = np.zeros(len(flagList_upupgo))

    for i in range(len(flagList_upupgo)):
        if flagList_doubleMA[i]==1 or flagList_upupgo[i]==1 or flagList_m[i]==1:
            flagList[i] = 1
        elif flagList_doubleMA[i]==-1 or flagList_upupgo[i]==-1 :
            flagList[i] = -1

    earningList, changePercent = test.test(flagList, data)
    print("changePercent:", changePercent)
    # utils.plotEarningRatio(earningList,flagList,data)
    return earningList[-1]

#github的tocken ：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
if __name__ == "__main__":
    stockCode =  "sh.600001"
    startDate = "2020-01-01"
    endDate = "2021-07-01"
    #testMA("./data/sh.600066.csv",startDate,endDate)

    folder = "./data"
    i=0
    chg_list = []
    for item in os.listdir(folder):
        file = os.path.join(folder,item)
        #chg = testMA(file, startDate, endDate)
        #chg = testMATrend(file, startDate, endDate,8)
        #chg = testMoment(file, startDate, endDate)
        #chg = testDoubleMA(file,startDate,endDate,fastMADays=7,slowMADays=14)
        #chg = testUpupGo(file,startDate,endDate)
        chg = testGroup(file,startDate,endDate)
        chg_list.append(chg)
        print("code:{},chg:{}".format(item,chg))

        i+=1
        if i>100:
            break
    print("mean:", np.mean(chg_list))
    print("median:", np.median(chg_list))
    utils.plotHist(chg_list)



#
# if __name__ == "__main__":
#     startDate = "2016-01-01"
#     endDate = "2021-07-01"
#     folder = "./data"
#     dp.downloadHS300(folder,startDate,endDate)


