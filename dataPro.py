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

    savePath = os.path.join(folder,stockCode+".csv")
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
    # 登录baostock
    lg = bs.login()
    print("login respond: lg.error_code={},lg.error_msg={}"
          .format(lg.error_code, lg.error_msg))

    # 获取沪深300成分股
    rs = bs.query_hs300_stocks()
    print('query_hs300 error_code:' + rs.error_code)
    print('query_hs300  error_msg:' + rs.error_msg)
    # 打印结果集
    hs300_stocks = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
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
        功能描述：
        @dateIndex 提取dataIndex前面的preDays的数据做特征，不包含datIndex天的数据
        @preDays   获取dateIndex前面的preDays天的数据，这样方便预测未来的结果
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


def downloadDataDemo():
    saveFolder = r"data"

    startDate = "2021-12-31"
    endDate = "2022-12-25"
    code = "sz.399300" # 沪深300指数
    data = downloadData(code,startDate,endDate)
    saveData(data,saveFolder,code)

def downloadHS300Demo():
    saveFolder = r"./data"

    startDate = "2021-12-31"
    endDate = "2022-12-25"
    downloadHS300(saveFolder, startDate, endDate)
    

if __name__ == "__main__":
    downloadDataDemo()
    # downloadHS300Demo()