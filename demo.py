import dataPro as dp
import strategy as sg
import test
import utils
import os

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
    flagList = sg.myMomentDot(data)
    earningList,changePercent = test.test(flagList,data)
    print("changePercent:",changePercent)
    #utils.plotEarningRatio(earningList,flagList,data)
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
        chg = testMoment(file, startDate, endDate)
        chg_list.append(chg)
        print("code:{},chg:{}".format(item,chg))

        i+=1
        if i>100:
            break
    utils.plotHist(chg_list)

#
# if __name__ == "__main__":
#     startDate = "2016-01-01"
#     endDate = "2021-07-01"
#     folder = "./data"
#     dp.downloadHS300(folder,startDate,endDate)


