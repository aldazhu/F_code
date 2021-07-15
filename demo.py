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
    flagList = sg.MAStrategy(data,MAdays)
    earningList,changePercent = test.test(flagList,data)
    print("changePercent:",changePercent)
    #utils.plotEarningRatio(earningList,flagList,data)
    return earningList[-1]



if __name__ == "__main__":
    stockCode =  "sh.600001"
    startDate = "2020-01-01"
    endDate = "2021-07-01"
    testMA("./data/sh.600066.csv",startDate,endDate)

    folder = "./data"

    for item in os.listdir(folder):
        file = os.path.join(folder,item)
        chg = testMA(file, startDate, endDate)
        print("code:{},chg:{}".format(item,chg))

