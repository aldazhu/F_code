import backtrader as bt
import numpy as np
import statsmodels.api as sm


class RSRS(bt.Indicator):
    lines = ('rsrs', 'R2')
    params = (('N', 18), ('value', 5))
    def __init__(self):
        self.high = self.data.high
        self.low = self.data.low
    def next(self):
        high_N = self.high.get(ago=0, size=self.p.N)
        low_N = self.low.get(ago=0, size=self.p.N)
        try:
            X = sm.add_constant(np.array(low_N))
            model = sm.OLS(np.array(high_N), X)
            results = model.fit()
            self.lines.rsrs[0] = results.params[1]
            self.lines.R2[0] = results.rsquared
        except:

            self.lines.rsrs[0] = 0

            self.lines.R2[0] = 0

class RSRS_Norm(bt.Indicator):
    lines = ('rsrs_norm','rsrs_r2','beta_right')
    params = (('N', 18), ('M', 200))
    def __init__(self):
        self.rsrs = RSRS(self.data)
        self.lines.rsrs_norm = (self.rsrs - bt.ind.Average(self.rsrs, period=self.p.M))/bt.ind.StandardDeviation(self.rsrs, period= self.p.M)
        self.lines.rsrs_r2 = self.lines.rsrs_norm * self.rsrs.R2
        self.lines.beta_right = self.rsrs * self.lines.rsrs_r2


class Diff(bt.Indicator):
    """current (close price - ema(close price, period)) / ema(close price, period)

    Args:
        bt (_type_): _description_
    """ 
    lines = ('diff',)
    params = (('ema_period', 30),)
    def __init__(self):
        self.emas = bt.indicators.EMA(self.data.close, period=self.p.ema_period)
        self.diff = (self.data.close - self.emas) / self.emas
    def next(self):
        self.lines.diff[0] = self.diff[0]

class AverageTrueRangeStop(bt.Indicator):
    lines = ('stop',)
    params = (('multiplier', 3), ('atr_period', 14), ('price_type', 'high'))
    def __init__(self):
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.highest = bt.indicators.Highest(self.data.high, period=self.p.atr_period)
        self.lowest = bt.indicators.Lowest(self.data.low, period=self.p.atr_period)
        self.atr_ema = bt.indicators.EMA(self.atr, period=self.p.atr_period)
        self.ema = bt.indicators.EMA(self.data.close, period=self.p.atr_period)
    def next(self):
        if self.p.price_type == 'close':
            self.lines.stop[0] = self.ema[0] - self.p.multiplier * self.atr_ema[0] # close [-1] is the last close price,cant use self.data.close[0] because it is the current close price
        elif self.p.price_type == 'low':
            self.lines.stop[0] = self.lowest[-1] - self.p.multiplier * self.atr_ema[0] # low [-1] is the last low price, cant use self.lowest[0] because it is the current low price
        elif self.p.price_type == 'high':
            self.lines.stop[0] = self.highest[0] - self.p.multiplier * self.atr_ema[0]
        else:
            self.lines.stop[0] = self.highest[0] - self.p.multiplier * self.atr_ema[0]
        # self.plotinfo.plotmaster = self.data  # Ensure the indicator is plotted on the main chart

class ATRNormalized(bt.Indicator):
    lines = ('atr_norm',)
    params = (('atr_period', 14), ('ema_period', 20))
    def __init__(self):
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.price_ema = bt.indicators.EMA((self.data.close + self.data.open) / 2.0, period=self.p.ema_period)
        self.atr_norm = self.atr / self.price_ema
    def next(self):
        self.lines.atr_norm[0] = self.atr_norm[0]

    
class SafeCCI(bt.indicators.CCI):
    lines = ('cci',)
    
    def __init__(self):
        super().__init__()

    def next(self):
        try:
            super().next()
            
        except ZeroDivisionError:
            
            self.lines.cci[0] = 0
        except Exception as e:
            print(e)
            self.lines.cci[0] = 0