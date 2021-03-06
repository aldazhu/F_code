import baostock as bs
import pandas as pd
import os
import numpy as np

# download data from Internet
def downloadData(stockCode:str,startDate:str,endDate:str,frequency:str="d"):
    """
    from Internet download data and return it.
    input paramters:
    stockCode : stock code eg: stock_code = "sh.600001"#
    startDate : eg:startDate = "2021-03-01"
    endDate : eg:endDate = "2021-06-01"
    frequency : optinal parameters: "h":hour,"d":day,"m":month

    output parameter:
    stockData: pandas format frame
    """
    lg = bs.login()
    print("login respond: lg.error_code={},lg.error_msg={}"
      .format(lg.error_code,lg.error_msg))
    itemList = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"
    rs = bs.query_history_k_data_plus(stockCode,itemList,
                                      startDate,endDate,
                                     frequency=frequency)

    dataList = []
    while(rs.error_code == "0" ) and rs.next():
        dataList.append(rs.get_row_data())
    stockData = pd.DataFrame(dataList,columns=rs.fields)
    return stockData


def saveData(data:"pandas frame",folder:"./data", stockCode:"sh.600001"):
    """
    input paramters:

    output paramters:
    None
    """

    savePath = os.path.join(folder,""+".csv")
    data.to_csv(savePath)
    return 


def getNormalData(data):
    """
    get " date open high low close volume amount " 

    input paramters:
    data : pandas dataFrame

    output paramters:
    pandas dataFrame
    """
    daily = {}
    daily["date"] = data["date"]
    daily["open"] = data["open"]
    daily["high"] = data["high"]
    daily["low"] = data["low"]
    daily["close"] = data["close"]
    daily["volume"] = data["volume"]
    daily["amount"] = data["amount"]

    d = pd.DataFrame(daily)
    d['date'] = pd.to_datetime(daily['date'])# use date as the index
    d.set_index(['date'],inplace=True)
    return d


def readData(filePath:"csv file path"):
    data = pd.read_csv(filePath)
    return data

def downloadHS300(folder:str,startDate:str,endDate:str,frequency:str="d"):

    if not os.path.isdir(folder):
        print(f"makedirs:{folder}")
        os.makedirs(folder)
    # ??????baostock
    lg = bs.login()
    print("login respond: lg.error_code={},lg.error_msg={}"
          .format(lg.error_code, lg.error_msg))

    # ????????????300?????????
    rs = bs.query_hs300_stocks()
    print('query_hs300 error_code:' + rs.error_code)
    print('query_hs300  error_msg:' + rs.error_msg)
    # ???????????????
    hs300_stocks = []
    while (rs.error_code == '0') & rs.next():
        # ?????????????????????????????????????????????
        hs300_stocks.append(rs.get_row_data())
    result = pd.DataFrame(hs300_stocks, columns=rs.fields)

    itemList = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"
    for stockCode in result["code"]:

        rs = bs.query_history_k_data_plus(stockCode, itemList,
                                          startDate, endDate,
                                          frequency=frequency)

        dataList = []
        while (rs.error_code == "0") and rs.next():
            dataList.append(rs.get_row_data())
        stockData = pd.DataFrame(dataList, columns=rs.fields)
        savePath = folder + "/{}.csv".format(stockCode)
        stockData.to_csv(savePath)
        print(savePath)


class dataPro():
    def __init__(self):
        self.data = []
        pass

    def readData(self,csvFilePath):
        self.data = pd.read_csv(csvFilePath)
        return self.data

    def getBatchData(self, dateIndex, preDays=5, futureDays = 1):
        '''
        ???????????????
        @dateIndex ??????dataIndex?????????preDays??????????????????????????????datIndex????????????
        @preDays   ??????dateIndex?????????preDays????????????????????????????????????????????????
        '''
        dataLen = len(self.data["open"])
        x = []
        y = (self.data['close'][dateIndex-1 + futureDays] - self.data['close'][dateIndex-1])/self.data['close'][dateIndex-1]
        for i in range(preDays):
            x.insert(0,self.data['open'][dateIndex -1 - i])
            x.insert(0,self.data['close'][dateIndex -1 - i])
            x.insert(0,self.data['high'][dateIndex -1 - i])
            x.insert(0,self.data['low'][dateIndex -1 - i])
            x.insert(0,self.data['close'][dateIndex -1 - i] - self.data['open'][dateIndex -1 - i])
            x.insert(0,np.log10(self.data['amount'][dateIndex -1 - i]))
            x.insert(0,self.data['turn'][dateIndex -1 - i])
        return np.array(x),np.array(y*100)

    def getHistoryData(self,startDate, endDate,preDays=5, futureDays=1):
        _x = []
        _y = []
        for i in range(startDate,endDate):
            x,y = self.getBatchData(i,preDays,futureDays)
            _x.insert(0,x)
            _y.insert(0,y)
        return np.array(_x),np.array(_y)


if __name__ == "__main__":
    saveFolder = r"./data_22"

    startDate = "2017-01-01"
    endDate = "2022-04-20"
    downloadHS300(saveFolder, startDate, endDate)