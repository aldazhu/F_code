import dataPro as dp
import utils
import numpy as np
import talib as ta


def MA(data,days:int):
    ma = [x for x in data["close"][:days]]
    for i in range(days,len(data["close"])):
        ma.append(sum(data["close"][i-days:i])/days)
    return ma

def EMA(data,days:int):
    ema = ta.EMA(data['close'],days)
    return ema

def CCI(data,days:int):
    '''
    Formula
    CCI = ( M - A ) / ( 0.015 * D )
    Where:
    M = ( H + L + C ) / 3
    H = Highest price for the period
    L = Lowest price for the period
    C = Closing price for the period
    A = n period moving average of M
    D = mean deviation of the absolute value of the difference between
        the mean price and the moving average of mean prices, M - A
    '''
    #CCI = [0 for i in range(days)]
    A = MA(data,days)
    M = (data['high'] + data['low'] + data['close']) / 3
    abs_tmp = abs(M-A)
    D = [1 for i in range(days)]
    for i in range(days,len(data['close'])):
        D.append(sum(abs_tmp[i-days:i]))
    M = np.array(M)
    D = np.array(D)
    A = np.array(A)
    CCI = (M - A) / (0.015 * D)#list不能直接做运算
    CCI[:days] = 0 #
    return CCI

def RSI(data,days:int):
    '''
    RSI = total Gain / (total Loss + total Gain)
    
    RS = total Gain / total Loss
    RSI = 100 - 100 / (1 + RS)
    '''
    #RSI = [0 for i in range(days)]
    gain = [0 for i in range(days)]
    loss = [0 for i in range(days)]
    for i in range(days,len(data['close'])):
        gain.append(sum([data['close'][i]-data['close'][i-1] for i in range(i-days,i) if data['close'][i]-data['close'][i-1]>0]))
        loss.append(sum([data['close'][i]-data['close'][i-1] for i in range(i-days,i) if data['close'][i]-data['close'][i-1]<0]))
    gain = np.array(gain)
    loss = np.array(loss)
    RS = gain/abs(loss+1e-6)
    RSI = 100 - 100 / (1 + RS)
    RSI[:days] = 50
    return RSI

def OSC(data,short:int,long:int):
    '''
    OSC = short EMA - long EMA
    '''
    short_ema = ta.EMA(data['close'],short)
    long_ema = ta.EMA(data['close'],long)
    OSC = short_ema - long_ema
    return OSC