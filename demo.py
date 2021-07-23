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
    flagList_moment = sg.myMomentDot(data)
    flagList_MATwoDay = sg.upMATwoDays(data,10)

    flagList = np.zeros(len(flagList_upupgo))

    for i in range(len(flagList_upupgo)):
        if flagList_doubleMA[i]==1 or flagList_upupgo[i]==1 or flagList_moment[i]==1 or flagList_MATwoDay[i]==1:
            flagList[i] = 1
        elif  flagList_upupgo[i]==-1 or flagList_doubleMA[i]==-1 :
            flagList[i] = -1

    earningList, changePercent = test.test(flagList, data)
    print("changePercent:", changePercent)
    # utils.plotEarningRatio(earningList,flagList,data)
    return earningList[-1]

def testdel(file,startDate,endDate):
    # data = dp.downloadData(stockCode,startDate,endDate)
    data = dp.readData(file)
    data = dp.getNormalData(data)
    flagList = sg.delmyMomentDot2(data)
    earningList, changePercent = test.test(flagList, data)
    print("changePercent:", changePercent)
    # utils.plotEarningRatio(earningList,flagList,data)
    return earningList[-1]


def testCovariance(folder,startDate,endDate):
    #当相关系数比较大时，是不是可以当做同一类
    files = os.listdir(folder)
    data = []
    name = []
    for i in range(len(files)):
        file = os.path.join(folder, files[i])
        data.append(dp.readData(file))
        name.append(file)

    for i in range(len(files)):
        data1 = data[i]
        for j in range(i,len(files)):
            data2=data[j]
            cov = utils.covariance(data1, data2)
            if cov>0.6 and cov<0.99:
                print("{},{},   cov={}".format(name[i],name[j],cov))
    return cov

if __name__ == "__main__":
    cov = testCovariance("./data"," "," ")
    print(cov)

#
# #github的tocken ：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
# if __name__ == "__main__":
#     stockCode =  "sh.600001"
#     startDate = "2020-01-01"
#     endDate = "2021-07-01"
#     #testMA("./data/sh.600066.csv",startDate,endDate)
#
#     folder = "./data"
#     i=0
#     chg_list = []
#     for item in os.listdir(folder):
#         file = os.path.join(folder,item)
#         #chg = testMA(file, startDate, endDate)
#         #chg = testMATrend(file, startDate, endDate,8)
#         #chg = testMoment(file, startDate, endDate)
#         #chg = testDoubleMA(file,startDate,endDate,fastMADays=7,slowMADays=14)
#         #chg = testUpupGo(file,startDate,endDate)
#         #chg = testGroup(file,startDate,endDate)
#         chg = testdel(file,startDate,endDate)
#
#         chg_list.append(chg)
#         print("code:{},chg:{}".format(item,chg))
#
#         i+=1
#         if i>100:
#             break
#     print("mean:", np.mean(chg_list))
#     print("median:", np.median(chg_list))
#     utils.plotHist(chg_list)



#
# if __name__ == "__main__":
#     startDate = "2016-01-01"
#     endDate = "2021-07-01"
#     folder = "./data"
#     dp.downloadHS300(folder,startDate,endDate)


