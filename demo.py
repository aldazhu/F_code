import dataPro as dp
import strategy as sg
import deepStrategy as dsg
import test
import utils
import os
import numpy as np

def testMA(file,startDate,endDate,MAdays=5):
    #ma均线策略，在ma均线上连续两天
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

def testUpMA(file,startDate,endDate,MAdays=5):
    #ma均线策略
    #data = dp.downloadData(stockCode,startDate,endDate)
    data = dp.readData(file)
    data = dp.getNormalData(data)
    flagList = sg.upMA(data,MAdays)
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

def testDoubleMA(file,startDate,endDate,fastMADays=5,slowMADays=15):
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
    flagList_MATwoDays = sg.upMATwoDays(data,10)
    # flagList_CCI = sg.CCIThresh(data,8,100,10)

    flagList = np.zeros(len(flagList_upupgo))

    for i in range(len(flagList_upupgo)):
        if flagList_doubleMA[i]==1 or flagList_upupgo[i]==1 or flagList_moment[i]==1 or flagList_MATwoDays[i]==1 \
                :
            flagList[i] = 1
        elif   flagList_doubleMA[i]==-1 or flagList_moment[i]==-1 :
            flagList[i] = -1

    earningList, changePercent = test.test(flagList, data)
    print("changePercent:", changePercent)
    #utils.plotEarningRatio(earningList[:100],flagList[:100],data[:100])
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

def testSvm(file):
    data = dp.readData(file)
    data = dp.getNormalData(data)
    svmp = dsg.svmPredict(data)
    flagList = svmp.getFlags()
    earningList, changePercent = test.test(flagList, data)
    print("changePercent:", changePercent)
    # utils.plotEarningRatio(earningList,flagList,data)
    return earningList[-1]

def testCCI(file,startDate,endDate,days=5,CCI_thresh=100,after_Ndays=20):
    """
    CCI< -thresh的 buy，大于 thresh的sell
    """
    data = dp.readData(file)
    data = dp.getNormalData(data)
    flagList = sg.CCIThresh(data,days,CCI_thresh,after_Ndays)
    earningList, changePercent = test.test(flagList, data)
    print("changePercent:", changePercent)
    # utils.plotEarningRatio(earningList,flagList,data)
    return earningList[-1]


def demo_testHS300():
    folder = "./data"
    startDate = "2020-01-01"
    endDate = "2021-01-01"
    i = 0
    chg_list = []
    for item in os.listdir(folder):
        file = os.path.join(folder, item)
        # chg = testMA(file, startDate, endDate)
        # chg = testMATrend(file, startDate, endDate,15)
        # chg = testUpMA(file, startDate, endDate,10)
        # chg = testMoment(file, startDate, endDate)
        # chg = testDoubleMA(file,startDate,endDate,fastMADays=7,slowMADays=14)
        # chg = testUpupGo(file,startDate,endDate)
        chg = testGroup(file, startDate, endDate)
        # chg = testdel(file,startDate,endDate)
        # chg = testSvm(file)
        # chg = testCCI(file,startDate,endDate,5,-100,10)

        chg_list.append(chg)
        print("code:{},chg:{}".format(item, chg))

        i += 1
        # if i>100:
        #     break
    print("mean:", np.mean(chg_list))
    print("median:", np.median(chg_list))
    utils.plotHist(chg_list)

def demo_testOneStock():
    filePath = "./data/sz.002773.csv"
    startDate = "2020-01-01"
    endDate = "2021-01-01"
    chg = testGroup(filePath, startDate, endDate)

#github的tocken ：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
if __name__ == "__main__":
    #demo_testOneStock()
    demo_testHS300()


#
# if __name__ == "__main__":
#     startDate = "2016-01-01"
#     endDate = "2021-07-01"
#     folder = "./data"
#     dp.downloadHS300(folder,startDate,endDate)


