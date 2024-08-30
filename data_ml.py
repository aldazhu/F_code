import os
import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from my_logger import logger

import talib as ta


class DataIndicator():
    def __init__(self) -> None:
        pass

    def get_indicator(self, file_path):
        data = pd.read_csv(file_path)
        indicators = {}
        indicators["MA5"] = ta.MA(data["close"], timeperiod=5)
        indicators["MA30"] = ta.MA(data["close"], timeperiod=30)

        # RSI
        indicators["RSI"] = ta.RSI(data["close"], timeperiod=14)

        # CCI
        indicators["CCI"] = ta.CCI(data["high"], data["low"], data["close"], timeperiod=14)

        # Stochastic Oscillator
        indicators["slowk"], indicators["slowd"] = ta.STOCH(data["high"], data["low"], data["close"], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

        # ADX
        indicators["ADX"] = ta.ADX(data["high"], data["low"], data["close"], timeperiod=14)

        # Momentum
        indicators["MOM"] = ta.MOM(data["close"], timeperiod=10)

        indicators["OBV"] = ta.OBV(data["close"], data["volume"])

        # MACD
        indicators["macd"], indicators["signal"], indicators["hist"] = ta.MACD(data["close"], fastperiod=12, slowperiod=26, signalperiod=9)

        # ATR
        indicators["ATR"] = ta.ATR(data["high"], data["low"], data["close"], timeperiod=14)

        # ROC
        indicators["ROC"] = ta.ROC(data["close"], timeperiod=10)
        
        return indicators
    
class IndictorDataset(Dataset):
    def __init__(self, csv_files, future_days=10) -> None:
        super().__init__()
        self.data = []
        self.indicator = []
        self.data_indicator = DataIndicator()
        self.data = []
        self.label = []
        for i, file in enumerate(csv_files):
            if not file.endswith(".csv"):
                continue
            if not os.path.isfile(file):
                logger.info(f"{file} is not a file")
                continue
            logger.info(f'{i} processing {file}')
            indicators = self.data_indicator.get_indicator(file)
            data = pd.read_csv(file)
            assert len(data["close"]) == len(indicators["MA5"])

            start_index = 0
            length = len(indicators["MA5"])
            for index in range(0, length):
                for key in indicators:
                    if np.isnan(indicators[key][index]) :
                        start_index += 1
                        break
            print(f"start_index: {start_index}")
            for key in indicators:
                indicators[key] = indicators[key][start_index:]

            # normalize the data
            indicators["MA5"] = (indicators["MA5"] ) / indicators["MA5"][start_index]
            indicators["MA30"] = (indicators["MA30"] ) / indicators["MA30"][start_index]
            indicators["OBV"] = (indicators["OBV"] - indicators["OBV"][start_index]) / indicators["OBV"][start_index]

            for i in range(start_index, length - future_days):
                x_i = []
                for key in indicators:
                    x_i.append(indicators[key][i])
                y_i = (data["close"][i+future_days] - data["close"][i]) / data["close"][i]
                if np.isnan(y_i) or np.isinf(y_i):
                    continue
                if np.isnan(x_i).any() or np.isinf(x_i).any():
                    continue
                self.data.append(x_i)
                self.label.append(y_i)
        self.data = np.array(self.data, dtype=np.float32)
        self.label = np.array(self.label, dtype=np.float32)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]

    

def demo_of_DataIndicator():
    file_path = 'data/sh.600009.csv'
    data_indicator = DataIndicator()
    indicators = data_indicator.get_indicator(file_path)
    for key in indicators:
        print(f"{key}: {indicators[key]}")
        print(f"{key} shape: {len(indicators[key])}")

def demo_of_IndictorDataset():
    csv_root = 'mini_data'
    csv_files = [os.path.join(csv_root, item) for item in os.listdir(csv_root)]
    indictor_dataset = IndictorDataset(csv_files)
    for i in range(len(indictor_dataset)):
        data, label = indictor_dataset[i]
        print(data)
        print(label)
        input("press any key to continue")
        

# get bach data
class MLDataTool():
    def __init__(self, pre_days, future_days):
        self.pre_days = pre_days
        self.future_days = future_days
        

    def get_shift_data(self, file_path):
        data = pd.read_csv(file_path)
        day_num = len(data["open"])

        X = []
        Y = []
        if day_num < self.pre_days + self.future_days:
            return X,Y

        for i in range(day_num - self.future_days - self.pre_days):
            # 提取某一特征self.pre_days天的数据，防止除以0 所以加一个很小的数“1e-9”(10的负九次方)
            # 涉及到价格的都以第一天的开盘价为0基准
            feat = lambda item: (data[item][i:i+self.pre_days] - data[item][i]) / data[item][i]  

            #x_i self.pre_days天的交易数据，以第一天的数据作为基准，后面几天的都用相对于第一天的百分比

            volumes = (data["volume"][i:i+self.pre_days] - data["volume"][i]) / (data["volume"][i] + 1e-9)
            turns = (data["turn"][i:i+self.pre_days] - data["turn"][i]) / (data["turn"][i] + 1e-9)
            
            x_i = [feat("open"),feat("high"),feat("low"),feat("close"),volumes,turns]
            # delete the nan data
            if np.isnan(x_i).any():
                continue
            y_i = (data["close"][i+self.pre_days: i+self.pre_days+self.future_days-1] - data["close"][i+self.pre_days-1] ) / data["close"][i+self.pre_days-1]
            if np.isnan(y_i).any():
                continue
            # x_x_array = [[day1_open,day1_high, day1_low,...],[day2_open,day2_high,day2_low,...],[day3_open,...]..]
            x_i_array = np.array(x_i, dtype=np.float32).T
            # 把x_i_array变成单行的向量
            item_num, days = x_i_array.shape[:2]
            # x_i_array = x_i_array.reshape(item_num*days)
            y_i_array = np.array(y_i, dtype=np.float32)
            X.append(x_i_array)
            Y.append(y_i_array)
        return X,Y
    

class MLDataset(Dataset):
    def __init__(self, csv_files, pre_days, future_days, use_catch=False, use_signal_future_day=True,npy_save_prefix="train") -> None:
        super().__init__()
        self.data = []
        self.label = []
        self.pre_days = pre_days
        self.future_days = future_days
        self.ml_data_tool = MLDataTool(pre_days, future_days)
        self.csv_files = []

        if use_catch:

            if os.path.exists(f"{npy_save_prefix}_data.npy") and os.path.exists(f"{npy_save_prefix}_label.npy"):
                print("loading data from the disk")
                self.data = np.load(f"{npy_save_prefix}_data.npy")
                self.label = np.load(f"{npy_save_prefix}_label.npy")
                return
        for i, file in enumerate(csv_files):
            if not file.endswith(".csv"):
                continue
            if not os.path.isfile(file):
                logger.info(f"{file} is not a file")
                continue
            logger.info(f'{i} processing {file}')
            X,Y = self.ml_data_tool.get_shift_data(file)
            self.data.extend(X)
            if use_signal_future_day:
                print(np.array(Y).shape)
                self.label.extend(np.array(Y)[:,-1])
            else:
                self.label.extend(Y)
            print(f"x shape: {np.array(X).shape}")
            print(f"y shape: {np.array(Y).shape}")
        # save the data to the disk
        self.data = np.array(self.data)
        self.label = np.array(self.label)
        print(f"data shape: {self.data.shape}")
        print(f"label shape: {self.label.shape}")
        np.save(f"{npy_save_prefix}_data.npy", self.data)
        np.save(f"{npy_save_prefix}_label.npy", self.label)

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    

def demo_of_MLDataset():
    csv_root = 'mini_data'
    pre_days = 15
    future_days = 10
    csv_files = [os.path.join(csv_root, item) for item in os.listdir(csv_root)]
    ml_dataset = MLDataset(csv_files, pre_days, future_days)
    for i in range(len(ml_dataset)):
        data, label = ml_dataset[i]
        print(data.shape)
        print(label.shape)
        break


def demo_of_MLDataTool():
    file_path = 'data/sh.600000.csv'
    pre_days = 15
    future_days = 10
    ml_data_tool = MLDataTool(pre_days, future_days)
    X,Y = ml_data_tool.get_shift_data(file_path)
    print(X[0])
    print(Y[0])




if __name__ == "__main__":
    # demo_of_MLDataTool()
    # demo_of_MLDataset()
    # demo_of_DataIndicator()
    demo_of_IndictorDataset()