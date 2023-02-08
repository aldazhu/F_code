import baostock as bs
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

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
        """Input hte path of the csv file ,return the data of csv in panda frame format

        Args:
            csvFilePath (str): the path of the csv

        Returns:
            panda frame : the stock data
        """
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
            x.insert(0,self.data['open'][dateIndex -1 - i])# 插入在最前面
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

class stockCluster():
    def __init__(self,preDays=5,featureDays=5) -> None:
        """_summary_

        Args:
            preDays (int, optional): 取今天前preDays的交易数据. Defaults to 5.
            featureDays (int, optional): 取featureDays天后相对于今天的变化百分比. Defaults to 5.
        """
        pass
        self.preDays_ = preDays
        self.featureDays_ = featureDays 

    def getShiftData(self,csvFilePath):
        """对于每一个样本点取前preDays天的数据（开盘价，最高价，最低价，收盘价，交易股数，换手率）做特征，每一个样本都用第一天的数据转为百分比值，
        例如：第一天的开盘价为10元，第二天的开盘价为10.1元，则第二天的开盘价变为，(10.1 - 10) / 10 = 0.01,第一天的开盘价变为(10-10)/10 = 0,
        self.featureDays_后close相对于今天的变化百分比
        返回类型：X=[    [day1_open,day1_high, day1_low,...,day2_open,day2_high,day2_low,...,day5_open,...],#样本1
                        [day2_open,day2_high, day2_low,...,day3_open,day4_high,day5_low,...,day6_open,...],#样本2
                        ...
                    ]
                Y=[
                    featuteDays_change,#样本1的标签
                    featuteDays+1_change,#样本2的标签
                    ...
                    ]
        
        Args:
            csvFilePath (str) : 股票文件的路径
           
        """
        data = pd.read_csv(csvFilePath)
        day_num = len(data["open"])

        X = []
        Y = []
        if day_num < self.preDays_ + self.featureDays_:
            return X,Y

        for i in range(day_num - self.featureDays_ - self.preDays_):
            # 提取某一特征self.preDays_天的数据，防止除以0 所以加一个很小的数“1e-9”(10的负九次方)
            # 涉及到价格的都以第一天的开盘价为0基准
            feat = lambda item: (data[item][i:i+self.preDays_] - data["open"][i]) / data["open"][i]  

            #x_i self.preDays_天的交易数据，以第一天的数据作为基准，后面几天的都用相对于第一天的百分比

            volumes = (data["volume"][i:i+self.preDays_] - data["volume"][i]) / (data["volume"][i] + 1e-9)
            turns = (data["turn"][i:i+self.preDays_] - data["turn"][i]) / (data["turn"][i] + 1e-9)
            x_i = [feat("open"),feat("high"),feat("low"),feat("close"),volumes,turns]
            y_i = (data["close"][i+self.preDays_+self.featureDays_-1] - data["close"][i+self.preDays_-1] ) / data["close"][i+self.preDays_-1]

            # x_x_array = [[day1_open,day1_high, day1_low,...],[day2_open,day2_high,day2_low,...],[day3_open,...]..]
            x_i_array = np.array(x_i).T
            # 把x_i_array变成单行的向量
            item_num, days = x_i_array.shape[:2]
            x_i_array = x_i_array.reshape(item_num*days)
            y_i_array = np.array(y_i)
            X.append(x_i_array)
            Y.append(y_i_array)
        return X,Y

    def drawOneSample(self,sample,y,if_save=False,save_path=""):
        """sample preDays天的（开盘价，最高价，最低价，收盘价，交易股数，换手率）组成的样本，
        [day1_open,day1_high, day1_low,...,day2_open,day2_high,day2_low,...,day5_open,...]

        Args:
            sample (_type_): _description_
            y (_type_): _description_
        """
        pass
        dates = np.arange(self.preDays_)
        # 每天的6项特征,open在第一位，每个6个数据是一个open价格
        opens = np.array(sample[::6]) 
        highs = np.array(sample[1::6])
        lows = np.array(sample[2::6])
        closes = np.array(sample[3::6])
        volumes = np.array(sample[4::6])
        turns = np.array(sample[5::6])

        colors_bool = closes >= opens
        colors = np.zeros(colors_bool.size, dtype="U5")
        colors[:] = "blue"
        colors[colors_bool] = "white"

        edge_colors = np.zeros(colors_bool.size, dtype="U1")
        edge_colors[:] = "b"
        edge_colors[colors_bool] = "r"

        plt.grid(True)
        # 画蜡烛图
        plt.bar(dates, (closes - opens), 0.8, bottom=opens, color=colors,edgecolor=edge_colors, zorder=3)
        plt.vlines(dates, lows, highs, color=edge_colors)

        # 画y
        datey = np.array([self.preDays_+self.featureDays_])
        preclose = sample[-3] # 样本最后一天的收盘价
        if y >= 0:
            colory = "r"
        else:
            colory = "b"
        plt.bar(datey,y,0.8,bottom=preclose,color=colory,edgecolor=colory,zorder=3)

        if if_save:
            if not save_path == "":
                print(f"save image to {save_path}")
                plt.savefig(save_path)
        else:
            plt.show()
        
        plt.clf()

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
    

def clusterDemo():
    src_root = r"./data"
    csv_file_path = r"./data/sh.600000.csv"

    preDays = 5
    featureDays = 5
    tool = stockCluster(preDays,featureDays)
    image_num = 0
    for csv_name in os.listdir(src_root):
        csv_file_path = os.path.join(src_root,csv_name)
        print(f"Find {csv_file_path}")
        if not os.path.isfile(csv_file_path):
            continue
        X,Y = tool.getShiftData(csv_file_path)
        for i in range(len(X)):
            image_num += 1
            image_save_path = f"temp/{str(image_num)}_pre{preDays}_feat{featureDays}.png"
            tool.drawOneSample(X[i],Y[i],True,image_save_path)


if __name__ == "__main__":
    pass
    clusterDemo()
    # downloadDataDemo()
    # downloadHS300Demo()