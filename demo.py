import dataPro as dp
import strategy as sg
import deepStrategy as dsg
import test
import utils
import os
import numpy as np

from my_logger import logger

def testMA(file,startDate,endDate,MAdays=5):
    #ma均线策略，在ma均线上连续两天
    #data = dp.downloadData(stockCode,startDate,endDate)
    data = dp.readData(file)
    data = dp.getNormalData(data)
    flagList = sg.upMATwoDays(data,MAdays)
    earningList,changePercent = test.test(flagList,data)
    print("MA changePercent:",changePercent)
    #utils.plotEarningRatio(earningList,flagList,data)
    return earningList[-1]

def testMATrend(file,startDate,endDate,MAdays=5):
    #ma均线策略
    #data = dp.downloadData(stockCode,startDate,endDate)
    data = dp.readData(file)
    data = dp.getNormalData(data)
    flagList = sg.MATrend(data,MAdays)
    earningList,changePercent = test.test(flagList,data)
    print("MAThread changePercent:",changePercent)
    #utils.plotEarningRatio(earningList,flagList,data)
    return earningList[-1]

def testUpMA(file,startDate,endDate,MAdays=5):
    #ma均线策略
    #data = dp.downloadData(stockCode,startDate,endDate)
    data = dp.readData(file)
    data = dp.getNormalData(data)
    flagList = sg.upMA(data,MAdays)
    earningList,changePercent = test.test(flagList,data)
    print("upMa changePercent:",changePercent)
    #utils.plotEarningRatio(earningList,flagList,data)
    return earningList[-1]

def testMoment(file,startDate,endDate):
    #data = dp.downloadData(stockCode,startDate,endDate)
    data = dp.readData(file)
    data = dp.getNormalData(data)
    # flagList = sg.myMomentDot2(data)
    flagList = sg.delmyMomentDot2(data)
    earningList,changePercent = test.test(flagList,data)
    print("Moment changePercent:",changePercent)
    #utils.plotEarningRatio(earningList,flagList,data)
    return earningList[-1]

def testDoubleMA(file,startDate,endDate,fastMADays=5,slowMADays=15):
    #data = dp.downloadData(stockCode,startDate,endDate)
    data = dp.readData(file)
    data = dp.getNormalData(data)
    flagList = sg.doubleMA(data,fastMADays,slowMADays)
    earningList,changePercent = test.test(flagList,data)
    print("double MA changePercent:",changePercent)
    #utils.plotEarningRatio(earningList,flagList,data)
    return earningList[-1]

def testUpupGo(file,startDate,endDate):
    #data = dp.downloadData(stockCode,startDate,endDate)
    data = dp.readData(file)
    data = dp.getNormalData(data)
    flagList = sg.upupgo(data)
    earningList,changePercent = test.test(flagList,data)
    print("up up go changePercent:",changePercent)
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
    flagList_CCI = sg.CCIThresh(data,9,90)
    flagList_RSI = sg.RSIStrategy(data, 9, 70, 30)
    flagList_OSC = sg.OSCStrategy(data,7,65)
    # log the flagList for each strategy
    logger.info("stock code: %s", file)
    logger.info("flagList_upupgo: buy num: %s, sell num :%s",np.sum(flagList_upupgo==1),np.sum(flagList_upupgo==-1))
    logger.info("flagList_doubleMA: buy num: %s, sell num :%s",np.sum(flagList_doubleMA==1),np.sum(flagList_doubleMA==-1))
    logger.info("flagList_moment: buy num: %s, sell num :%s",np.sum(flagList_moment==1),np.sum(flagList_moment==-1))
    logger.info("flagList_MATwoDays: buy num: %s, sell num :%s",np.sum(flagList_MATwoDays==1),np.sum(flagList_MATwoDays==-1))
    logger.info("flagList_CCI: buy num: %s, sell num :%s",np.sum(flagList_CCI==1),np.sum(flagList_CCI==-1))
    logger.info("flagList_RSI: buy num: %s, sell num :%s",np.sum(flagList_RSI==1),np.sum(flagList_RSI==-1))
    logger.info("flagList_OSC: buy num: %s, sell num :%s",np.sum(flagList_OSC==1),np.sum(flagList_OSC==-1))

    flagList = np.zeros(len(flagList_upupgo))

    flagList = flagList_RSI + flagList_doubleMA + flagList_CCI
    flagList = np.where(flagList>=1,1,flagList)
    flagList = np.where(flagList<=-1,-1,flagList)
    logger.info("final flagList: buy num: %s, sell num :%s",np.sum(flagList==1),np.sum(flagList==-1))

        
    earningList, changePercent = test.test(flagList, data)
    logger.info("group changePercent: %s", changePercent)
    #utils.plotEarningRatio(earningList[:100],flagList[:100],data[:100])
    return earningList[-1]

def test_RSI(file,startDate,endDate):
    data = dp.readData(file)
    data = dp.getNormalData(data)
    days = 14
    high_thresh = 70
    low_thresh = 30
    flagList = sg.RSIStrategy(data, days, high_thresh, low_thresh)
    earningList, changePercent = test.test(flagList, data)
    print("RSI changePercent:", changePercent)
    # utils.plotEarningRatio(earningList,flagList,data)
    return earningList[-1]


def testOSC(file,startDate,endDate,short=5,long=10):
    data = dp.readData(file)
    data = dp.getNormalData(data)
    flagList = sg.OSCStrategy(data,short,long)
    earningList, changePercent = test.test(flagList, data)
    print("OSC changePercent:", changePercent)
    # utils.plotEarningRatio(earningList,flagList,data)
    return changePercent

def test_TrendFollowingStrategy(file,startDate,endDate):
    data = dp.readData(file)
    data = dp.getNormalData(data)
    days = 5
    flagList = sg.TrendFollowingStrategy(data, days)
    earningList, changePercent = test.test(flagList, data)
    print("TrendFollowingStrategy changePercent:", changePercent)
    # utils.plotEarningRatio(earningList,flagList,data)
    return earningList[-1]


def testCovariance(folder):
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
            # if cov>0.8 and cov<0.9999:
            #     print("{},{},   cov={}".format(name[i],name[j],cov))
            if cov < -0.2:
                print("{},{},   cov={}".format(name[i], name[j], cov))
    return cov

def testSvm(file):
    data = dp.readData(file)
    data = dp.getNormalData(data)
    svmp = dsg.svmPredict(data)
    flagList = svmp.getFlags()
    earningList, changePercent = test.test(flagList, data)
    print("SVM changePercent:", changePercent)
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
    print("CCI changePercent:", changePercent)
    # utils.plotEarningRatio(earningList,flagList,data)
    return earningList[-1]


def demo_testHS300():
    folder = "./data"
    startDate = "2020-01-01"
    endDate = "2024-02-05"
    i = 0
    chg_list = []
    stock_code_list = []
    stock_state_dict = {}
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
        # chg = test_RSI(file,startDate,endDate)
        # chg = test_TrendFollowingStrategy(file,startDate,endDate)
        # chg = testOSC(file,startDate,endDate,5,10)

        chg_list.append(chg)
        
        print("code:{},chg:{}".format(item, chg))
        stock_state_dict[item] = chg

        i += 1
        # if i>100:
        #     break
    for k, v in stock_state_dict.items():
        print(f"stock code: {k}, change: {v}")
    print("mean:", np.mean(chg_list))
    print("median:", np.median(chg_list))
    print("standard variance:", np.std(chg_list))
    print("max:", np.max(chg_list))
    print("min:", np.min(chg_list))
    utils.plotHist(chg_list)

def demo_testOneStock():
    filePath = "./data/sh.600000.csv"
    startDate = "2020-01-01"
    endDate = "2024-02-05"
    chg = testGroup(filePath, startDate, endDate)
    chg = testMA(filePath, startDate, endDate)
    chg = testMATrend(filePath, startDate, endDate,15)
    chg = testUpMA(filePath, startDate, endDate,10)
    chg = testMoment(filePath, startDate, endDate)
    chg = testDoubleMA(filePath,startDate,endDate,fastMADays=7,slowMADays=14)
    chg = test_TrendFollowingStrategy(filePath,startDate,endDate)


def demo_of_covariance():
    folder = "./data"
    testCovariance(folder)


#github的tocken ：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
if __name__ == "__main__":
    demo_testOneStock()
    # demo_of_covariance()
    # demo_testHS300()
    # delf()

