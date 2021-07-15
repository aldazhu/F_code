import baostock as bs
import pandas as pd
import os

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
