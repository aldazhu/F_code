import baostock as bs
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


class DataPro():
    def __init__(self):
        self.data = []
        self.is_baostock_init = False
        self.save_root = None
        self.item_dict = {
            "stock_day_history": "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTm,pbMRQ,psTTM,pcfNcfTTM,isST",
            "stock_minute_history": "date,time,code,open,high,low,close,volume,amount,adjustflag",
            "index_history": "date,code,open,high,low,close,volume,amount,adjustflag",
            "etf_history": "date,code,open,high,low,close,preclose,volume,amount,pctChg,adjustflag"
        }
        pass

    def set_save_root(self, save_root):
        self.save_root = save_root

    def save_data(self, data, stock_code):
        if self.save_root is None:
            raise Exception("save root is None, please set the save root first.")

        if not os.path.isdir(self.save_root):
            os.makedirs(self.save_root)
        save_path = os.path.join(self.save_root, stock_code + ".csv")
        data.to_csv(save_path)
        print(f"save data to {save_path}")
        

    def init_baostock(self):
        lg = bs.login()
        print("login respond: lg.error_code={},lg.error_msg={}"
              .format(lg.error_code, lg.error_msg))
        if lg.error_code == "0":
            self.is_baostock_init = True
        return self.is_baostock_init

    def download(self, stock_code, start_date, end_date, item_list, frequency,adjustflag="2"):
        """
        args:
        stock_code : stock code eg: stock_code = "sh.600001"
        start_date : eg:startDate = "2021-03-01"
        end_date : eg:endDate = "2021-06-01"
        item_list : the list of the items
        frequency : optinal parameters: "h":hour,"d":day,"w": week, "m":month, "60":60 minute, "30":30 minute, "15":15 minute, "5":5 minute
        adjustflag : 1 后复权， 2：前复权， 3：不复权
        """
        if not self.is_baostock_init:
            assert self.init_baostock() == True, "baostock init failed"
        
        rs = bs.query_history_k_data_plus(stock_code, item_list,
                                          start_date, end_date,
                                          frequency=frequency,
                                          adjustflag=adjustflag)
        if rs.error_code != "0":
            print(f"download {stock_code} data failed")
            print(f"error_code:{rs.error_code}, error_msg:{rs.error_msg}")
            return None
        dataList = []
        while (rs.error_code == "0") and rs.next():
            dataList.append(rs.get_row_data())
        stockData = pd.DataFrame(dataList, columns=rs.fields)
        return stockData

    def download_k_history(self, stock_code, start_date, end_date,frequency="d",adjustflag="2", mode="stock"):
        """
        stock_code : stock code eg: stock_code = "sh.600001"
        start_date : eg:startDate = "2021-03-01"
        end_date : eg:endDate = "2021-06-01"
        frequency : optinal parameters: "h":hour,"d":day,"w": week, "m":month, "60":60 minute, "30":30 minute, "15":15 minute, "5":5 minute
        adjustflag : 1 后复权， 2：前复权， 3：不复权
        mode : optinal parameters: "stock","index","etf", index has no minute data
        """
        pass
        if mode == "stock":
            if frequency == "d" or frequency == "w" or frequency == "m":
                item_list = self.item_dict["stock_day_history"]
            elif frequency == "60" or frequency == "30" or frequency == "15" or frequency == "5":
                item_list = self.item_dict["stock_minute_history"]
            else:
                raise Exception("frequency is not supported")
        
        elif mode == "index":
            item_list = self.item_dict["index_history"]
        elif mode == "etf":
            item_list = self.item_dict["etf_history"]

        stock_data = self.download(stock_code, start_date, end_date, item_list, frequency, adjustflag)
        self.save_data(stock_data, stock_code)        
        print(f"download {stock_code} data success")

    def download_all_hs300_stocks(self, start_date, end_date, frequency="d"):
        """
        start_date : eg:startDate = "2021-03-01"
        end_date : eg:endDate = "2021-06-01"
        frequency : optinal parameters: "h":hour,"d":day,"w": week, "m":month, "60":60 minute, "30":30 minute, "15":15 minute, "5":5 minute
        """
        pass
        if not self.is_baostock_init:
            assert self.init_baostock() == True, "baostock init failed"
        rs = bs.query_hs300_stocks()
        print('query_hs300 error_code:' + rs.error_code)
        print('query_hs300  error_msg:' + rs.error_msg)
        hs300_stocks = []
        while (rs.error_code == '0') & rs.next():
            hs300_stocks.append(rs.get_row_data())
        
        count = 0
        with open("names.txt", "w") as f:
            for stock in hs300_stocks:
                stock_code = stock[1]
                stock_name = stock[2]
                f.write(stock_code + " " + stock_name + "\n")
                
                self.download_k_history(stock_code, start_date, end_date, frequency)

    def get_market(self, stock_code):
        """Determine which market the stock belongs to"""
        if stock_code.startswith('60'):
            return 'Shanghai'
        elif stock_code.startswith('00') or stock_code.startswith('30'):
            return 'Shenzhen'
        elif stock_code.startswith('68'):
            # Shanghai Stock Exchange Science and Technology Innovation Board
            return 'STAR'
        else:
            return 'Unknown Market'

    def download_all_zh1000_stocks(self, start_date, end_date, frequency="d",zh1000_names_path="doc/names_zh1000.txt"):
        """read doc/names_zh1000.txt and download the stocks

        Args:
            start_date (_type_): _description_
            end_date (_type_): _description_
            frequency (str, optional): _description_. Defaults to "d".
            zh1000_names_path ()

        Returns:
            _type_: _description_
        """
        with open(zh1000_names_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        current_num = 0
        skip_num = 0
        for line in lines:
            stock_code , stock_name = line.strip().split(' ')
            current_num += 1
            print(f"{current_num}/{total_lines} skipnum:{skip_num} \t {stock_code} {stock_name}")
            market = self.get_market(stock_code)
            if market == "STAR":
                print(" skip STAR code")
                skip_num += 1
                continue
            self.download_k_history(stock_code, start_date, end_date, frequency)

            

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
    itemList = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTm,pbMRQ,psTTM,pcfNcfTTM,isST"
    rs = bs.query_history_k_data_plus(stockCode,itemList,
                                      startDate,endDate,
                                     frequency=frequency,
                                     adjustflag=2)
    # adjustflag : 1 后复权， 2：前复权， 3：不复权

    dataList = []
    while(rs.error_code == "0" ) and rs.next():
        dataList.append(rs.get_row_data())
    stockData = pd.DataFrame(dataList,columns=rs.fields)
    return stockData

def download_index_data(indexCode:str,startDate:str,endDate:str,frequency:str="d"):
    lg = bs.login()
    print("login respond: lg.error_code={},lg.error_msg={}"
      .format(lg.error_code,lg.error_msg))
    itemList = "date,code,open,high,low,close,preclose,volume,amount,pctChg"
    rs = bs.query_history_k_data_plus(indexCode,itemList,
                                      startDate,endDate,
                                     frequency=frequency)
    # adjustflag : 1 后复权， 2：前复权， 3：不复权

    dataList = []
    while(rs.error_code == "0" ) and rs.next():
        dataList.append(rs.get_row_data())
    stockData = pd.DataFrame(dataList,columns=rs.fields)
    return stockData

def download_etf(stockCode:str,startDate:str,endDate:str,frequency:str="d"):
    lg = bs.login()
    print("login respond: lg.error_code={},lg.error_msg={}"
      .format(lg.error_code,lg.error_msg))
    itemList = "date,code,open,high,low,close,preclose,volume,amount,adjustflag"
    rs = bs.query_history_k_data_plus(stockCode,itemList,
                                      startDate,endDate,
                                     frequency=frequency,
                                     adjustflag="2")

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
    if not os.path.isdir(folder):
        os.makedirs(folder)

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

    itemList = "date,time,code,open,high,low,close,volume,amount,adjustflag"
    for stockCode in result["code"]:

        rs = bs.query_history_k_data_plus(stockCode, itemList,
                                          startDate, endDate,
                                          frequency=frequency,
                                          adjustflag="2")

        dataList = []
        while (rs.error_code == "0") and rs.next():
            dataList.append(rs.get_row_data())
        stockData = pd.DataFrame(dataList, columns=rs.fields)
        savePath = folder + "/{}.csv".format(stockCode)
        stockData.to_csv(savePath)
        print(savePath)


def get_hs300_profit_data(folder:str,startDate:str,endDate:str):
    """
    get the profit of hs300

    input paramters:
    folder : the folder of the stock data
    startDate : the start date of the data
    endDate : the end date of the data

    output paramters:
    profit : the profit of the hs300
    """
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
    for stock in hs300_stocks:
        rs = bs.query_profit_data(code=stock[1], year=2021, quarter=3)

    
    return 

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
        # dataLen = len(self.data["open"])
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
    
    def getNormalizeBatchData(self, dateIndex, preDays=5, futureDays = 1):
        '''
        功能描述：
        @dateIndex 提取dataIndex前面的preDays的数据做特征，不包含datIndex天的数据
        @preDays   获取dateIndex前面的preDays天的数据，这样方便预测未来的结果
        '''
        xi, yi = self.getBatchData(dateIndex, preDays, futureDays)
        xi = (xi - np.mean(xi)) / np.std(xi)
        return xi, yi


    def getHistoryData(self,startDate, endDate,preDays=5, futureDays=1, normalize=False):
        _x = []
        _y = []
        for i in range(startDate,endDate):
            if normalize:
                x,y = self.getNormalizeBatchData(i,preDays,futureDays)
            else:
                x,y = self.getBatchData(i,preDays,futureDays)
            _x.insert(0,x)
            _y.insert(0,y)
        return np.array(_x),np.array(_y)

class stockCluster():
    def __init__(self,preDays=5,futureDays=5) -> None:
        """_summary_

        Args:
            preDays (int, optional): 取今天前preDays的交易数据. Defaults to 5.
            futureDays (int, optional): 取futureDays天后相对于今天的变化百分比. Defaults to 5.
        """
        pass
        self.preDays_ = preDays
        self.futureDays_ = futureDays 

    def getShiftData(self,csvFilePath):
        """对于每一个样本点取前preDays天的数据（开盘价，最高价，最低价，收盘价，交易股数，换手率）做特征，每一个样本都用第一天的数据转为百分比值，
        例如：第一天的开盘价为10元，第二天的开盘价为10.1元，则第二天的开盘价变为，(10.1 - 10) / 10 = 0.01,第一天的开盘价变为(10-10)/10 = 0,
        self.futureDays_后close相对于今天的变化百分比
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
        if day_num < self.preDays_ + self.futureDays_:
            return X,Y

        for i in range(day_num - self.futureDays_ - self.preDays_):
            # 提取某一特征self.preDays_天的数据，防止除以0 所以加一个很小的数“1e-9”(10的负九次方)
            # 涉及到价格的都以第一天的开盘价为0基准
            feat = lambda item: (data[item][i:i+self.preDays_] - data["open"][i]) / data["open"][i]  

            #x_i self.preDays_天的交易数据，以第一天的数据作为基准，后面几天的都用相对于第一天的百分比

            volumes = (data["volume"][i:i+self.preDays_] - data["volume"][i]) / (data["volume"][i] + 1e-9)
            turns = (data["turn"][i:i+self.preDays_] - data["turn"][i]) / (data["turn"][i] + 1e-9)
            x_i = [feat("open"),feat("high"),feat("low"),feat("close"),volumes,turns]
            y_i = (data["close"][i+self.preDays_+self.futureDays_-1] - data["close"][i+self.preDays_-1] ) / data["close"][i+self.preDays_-1]

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
        datey = np.array([self.preDays_+self.futureDays_])
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
    saveFolder = r"data_index"

    startDate = "2016-1-1"
    endDate = "2024-12-25"
    code = "sh.000300" # 沪深300指数
    # data = downloadData(code,startDate,endDate)
    data = download_index_data(code,startDate,endDate)
    saveData(data,saveFolder,code)

def downloadHS300Demo():
    saveFolder = r"./data_hour_hs300"

    startDate = "2022-01-01"
    endDate = "2024-07-05"
    frequency = "60"
    downloadHS300(saveFolder, startDate, endDate, frequency)

def downloadZZ1000Demo():
    save_folder = r"./data/zz1000"
    start_date = "2020-01-01"
    end_date = "2023-05-05"
    downloadData()
    
def demo_of_download_etf():
    save_folder = r"./data_etf"
    start_date = "2020-01-01"
    end_date = "2023-05-05"
    etf_list = ["sh.510300","sh.510500","sh.510050","sh.510180","sh.510310","sh.510330","sh.510900","sh.510880","sh.510660","sh.510230","sh.510270","sh.510210"]
    for etf in etf_list:
        data = download_etf(etf, start_date, end_date)
        saveData(data, save_folder, etf)

def clusterDrawDemo():
    src_root = r"./data"
    csv_file_path = r"./data/sh.600000.csv"

    preDays = 5
    futureDays = 5
    tool = stockCluster(preDays,futureDays)
    image_num = 0
    for csv_name in os.listdir(src_root):
        csv_file_path = os.path.join(src_root,csv_name)
        print(f"Find {csv_file_path}")
        if not os.path.isfile(csv_file_path):
            continue
        X,Y = tool.getShiftData(csv_file_path)
        for i in range(len(X)):
            image_num += 1
            image_save_path = f"temp/{str(image_num)}_pre{preDays}_feat{futureDays}.png"
            tool.drawOneSample(X[i],Y[i],True,image_save_path)

def clusterSaveSampleDemo():
    src_root = r"./data"
    
    sample_save_path = r"G:\pydoc\sample"
    if not os.path.isdir(sample_save_path):
        os.makedirs(sample_save_path)

    preDays = 5
    futureDays = 5
    tool = stockCluster(preDays,futureDays)
    save_x = []
    save_y = []
    for csv_name in os.listdir(src_root):
        csv_file_path = os.path.join(src_root,csv_name)
        print(f"Find {csv_file_path}")
        if not os.path.isfile(csv_file_path):
            continue
        X,Y = tool.getShiftData(csv_file_path)
        save_x.extend(X)
        save_y.extend(Y)
    save_x = np.array(save_x)
    save_y = np.array(save_y)
    np.save(f"{sample_save_path}\\samples.npy",save_x)
    np.save(f"{sample_save_path}\\labels.npy",save_y)


class ClusterSamples():
    def __init__(self,cluster_num,max_iterate_times, min_change_rate=0.01):
        """_summary_

        Args:
            cluster_num (int): 聚类中心数目
            max_iterate_times (int): 迭代次数, max_iterate_times
        """
        pass
        self.k = cluster_num
        self.times = max_iterate_times
        self.min_change_rate = min_change_rate
        self.current_change_rate = 1 # 当前的变化率,初始化为1,表示全部都要更新
        self.best_centers = None
        self.best_labels = None
        self.stop_iterate = False
        print(f"cluster_num:{self.k},max_iterate_times:{self.times},min_change_rate:{self.min_change_rate}")
        
    
    def loadSamples(self,samples_path,labels_path):
        """输入numpy保存的样本和标签路径读取样本和标签

        Args:
            samples_path (str): _description_
            labels_path (_type_): _description_
        """
        X = np.load(samples_path,allow_pickle=True)
        Y = np.load(labels_path,allow_pickle=True)
        return X,Y

    def cluster(self,X,centers,old_labels=None):
        """聚类

        Args:
            X (_type_): 数据样本，
            labels (): X的类别
            centers (_type_): 聚类中心
        """
        print(f"Clustering \t clusters number: {len(centers)}")
        labels = []
        for i in range(len(X)):
            min_distance = 99999999999999
            label = 0
            for j in range(len(centers)): # 找到离x[i]最近的中心
                distance = np.sum((centers[j] - X[i])**2)
                if distance < min_distance:
                    label = j
                    min_distance = distance
            labels.append(label)
        if old_labels is not None:
            change_rate = np.sum(np.array(labels) != np.array(old_labels)) / len(X)
            print(f"samples number:{len(X)}, changed number:{np.sum(np.array(labels) != np.array(old_labels))}")
            print(f"labels change rate:{change_rate}")
            if change_rate < self.current_change_rate:
                self.best_centers = centers
                self.best_labels = labels
                self.current_change_rate = change_rate
            if change_rate < self.min_change_rate:
                self.stop_iterate = True
        return labels
    
    def update(self,X,labels):
        """根据labels重新计算聚类中心

        Args:
            X (_type_): _description_
            labels (_type_): _description_
        """
        print("update the cluster centers")
        assert(len(X) == len(labels))
        centers_dict = {} # item 是 label:[X[i],num]
        for i in range(len(X)):
            if labels[i] in centers_dict:
                centers_dict[labels[i]][0] += X[i]
                centers_dict[labels[i]][1] += 1
            else:
                centers_dict[labels[i]] = [X[i],1]
        
        centers = []
        for item in centers_dict.items():
            center = item[1][0] / item[1][1] 
            centers.append(center)
        return centers
    
    def initialize(self,X):
        """初始化聚类中心

        Args:
            X (_type_): _description_
        """
        print("Initialize the cluster center")
        step = len(X) // self.k # 向下取整
        centers = []
        for i in range(0,len(X),step):
            centers.append(X[i])
        return centers

 
    def run(self,X):
        centers = self.initialize(X)
        old_labels = None
        for t in range(self.times):
            print(f"run times:{t}")
            labels = self.cluster(X,centers,old_labels)
            centers = self.update(X,labels)
            old_labels = labels
            if self.stop_iterate:
                break
        return X,labels,centers
        
    def save(self,X,labels,centers):
        """_summary_

        Args:
            X (_type_): _description_
            labels (_type_): _description_
            centers (_type_): _description_
        """
        pass

def DemoOfCluster():
    samples_path = r"sample/samples.npy"
    labels_path = r"sample/labels.npy"
    save_root = r"temp"
    max_iter_num = 50
    cluster_num = 20
    solu = ClusterSamples(cluster_num,max_iter_num)
    X,Y = solu.loadSamples(samples_path,labels_path)
    X,labels,centers = solu.run(X)
    
    # 保存数据
    preDays = 5
    futureDays = 5
    tool = stockCluster(preDays,futureDays)
    image_num = 0
    for i in range(len(X)):
        image_num += 1
        image_save_folder = os.path.join(save_root,str(labels[i]))
        if not os.path.isdir(image_save_folder):
            os.makedirs(image_save_folder)

        image_save_path =os.path.join(image_save_folder,f"{str(image_num)}_pre{preDays}_feat{futureDays}.png") 
        tool.drawOneSample(X[i],Y[i],True,image_save_path)

def demo_of_datapro():
    save_root = "./data_test"
    dp = DataPro()
    dp.set_save_root(save_root)
    stock_code = "sh.000300"
    start_date = "2022-01-01"
    end_date = "2024-09-01"
    frequency = "d"
    # dp.download_k_history(stock_code, start_date, end_date, frequency,mode="index")
    # dp.download_k_history(stock_code, start_date, end_date, frequency,mode="stock")
    dp.download_all_hs300_stocks(start_date, end_date, frequency)
    # dp.download_all_zh1000_stocks(start_date,end_date, frequency)


def demo_of_filter_zero_data():
    pass
    root = "./data"
    # filter the data, nan or 0.0
    total_file_num = len(os.listdir(root))
    filter_file_num = 0
    items = ["open","high","low","close"]
    for file_name in os.listdir(root):
        file_path = os.path.join(root, file_name)
        print(f"processing {file_path}")
        data = pd.read_csv(file_path)
        src_size = len(data)
        # filter the 0 in data columns
        for item in items:
            print("filter item:",item)
            for i in range(len(data[item])):
                # print(abs(data[item][i] ))
                # break

                if abs(data[item][i] ) < 1e-6:
                    print(f"filter {file_path} {item} {i}")
                    data[item][i] = data[item][i-1]

        print(f"total file num:{total_file_num}, filter file num:{filter_file_num}")
        # break
        
def demo_of_write_zh1000_to_txt():
        
    # 读取CSV文件
    df = pd.read_csv('zh_1000.CSV', encoding='gbk')

    # 提取股票代码和股票名称
    stock_codes = df['股票代码']
    stock_names = df['股票简称']

    # 打印结果
    # print(stock_codes)
    # print(stock_names)

    with open('doc/names_zh1000.txt', 'w', encoding='utf-8') as f:
        for stock_code , stock_name in zip(stock_codes, stock_names):
            f.write(f"{stock_code} {stock_name} \n" )


def demo_of_analysis_data():
    pass
    root = "./data_zh1000"
    item_num_set = set()
    filter_num = 0 
    for file_name in os.listdir(root):
        file_path = os.path.join(root, file_name)
        print(f"processing {file_path}")
        data = pd.read_csv(file_path)
        item_num_set.add(len(data['open']))
        if len(data['open']) < 700:
            filter_num += 1
            print(f"{filter_num} less than 1000 {file_path}")
            print(data)
    print(item_num_set)
    print(f"total file num:{len(item_num_set)}, min:{min(item_num_set)}, max:{max(item_num_set)}")

def demo_of_get_stock_value():


    # 登录系统
    lg = bs.login()

    # 获取股票基本信息
    rs = bs.query_stock_basic(code="sh.600000")

    # 打印结果
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)

    # 登出系统
    bs.logout()

    print(result)


if __name__ == "__main__":
    pass
    # clusterDrawDemo()
    # clusterSaveSampleDemo()
    # downloadDataDemo()
    # demo_of_download_etf()
    # downloadHS300Demo()
    # DemoOfCluster()
    # demo_of_datapro()
    # demo_of_filter_zero_data()
    demo_of_datapro()
    # demo_of_write_zh1000_to_txt()
    # demo_of_analysis_data()
    # demo_of_get_stock_value()