import backtrader as bt
import datetime
import random
import pandas as pd

import os

from backtrader_strategy import *


def get_valid_files(data_root, start_date, end_date):
    """
    get the valid files in the data_root, which is between start_date and end_date
    """
    valid_files = []
    for item in os.listdir(data_root):
        file = os.path.join(data_root, item)
        df = pd.read_csv(file)
        date_key = 'date'
        if date_key not in df.columns and 'Date' in df.columns:
            date_key = 'Date'
            
        df[date_key] = pd.to_datetime(df[date_key])
        print(f"Processing {file}, min date: {df[date_key].min()}, max date: {df[date_key].max()}")
        if df[date_key].min() <= start_date :
            valid_files.append(file)
    print(f"Total valid files: {len(valid_files)}")
    return valid_files

def test_backtrader(datas, strategies, cash=100000.0, commission=0.001,stake=100, visual_data=False):
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    for strategy in strategies:
        cerebro.addstrategy(strategy)
        # cerebro.optstrategy(strategy, rsi_period=range(10, 31))

    # Add the Data Feed to Cerebro
    for data in datas:
        cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(cash)

    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.FixedSize, stake=stake)

    # Set the commission
    cerebro.broker.setcommission(commission=commission)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='mysharpe', riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns)

    # Run over everything
    results = cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    print('Sharpe Ratio:', results[0].analyzers.mysharpe.get_analysis())
    print('max Draw Down:', results[0].analyzers.drawdown.get_analysis()['max'])
    print('return:', results[0].analyzers.returns.get_analysis()['rnorm100'])

    
    # Print the profit
    profit = cerebro.broker.getvalue() - cash

    # Visulize the result
    if visual_data:
        cerebro.plot()

    return profit


def get_data(data_name, from_date, to_date):
    if not os.path.exists(data_name):
        print(f"Data file {data_name} does not exist.")
        raise "Data file does not exist."

    data = MyData(dataname=data_name,
                fromdate=from_date,
                todate=to_date)
    return data

def get_minutely_data(data_name, from_date, to_date):
    if not os.path.exists(data_name):
        print(f"Data file {data_name} does not exist.")
        raise "Data file does not exist."

    data = MyMinutelyData(dataname=data_name,
                fromdate=from_date,
                todate=to_date)
    return data

def get_sea_data(data_name, from_date, to_date):
    if not os.path.exists(data_name):
        print(f"Data file {data_name} does not exist.")
        raise "Data file does not exist."

    data = MySeaData(dataname=data_name,
                fromdate=from_date,
                todate=to_date)
    return data


def demo_of_ShortTermReversalEffectinStocks():
    data_root = 'data'
    from_date = datetime.datetime(2016, 1, 1)
    to_date = datetime.datetime(2024, 12, 31)
    data_names = [os.path.join(data_root, item) for item in os.listdir(data_root)]
    if "hour" in data_names[0]:
        datas = [get_minutely_data(data_name, from_date, to_date) for data_name in data_names]
    else:
        datas = [get_data(data_name, from_date, to_date) for data_name in data_names]

    stake = 100

    strategies = [
        ShortTermReversalEffectinStocks
    ]

    test_backtrader(datas, strategies=strategies, cash=100000.0, commission=0.001, stake=stake)


def demo_of_simple_strategy():
    # Create a Data Feed
    data_root = "data"
    # data_root = "data_train"
    test_all_data = True
    from_date = datetime.datetime(2022, 1, 5)
    to_date = datetime.datetime(2024, 1, 30)
    cash = 10000

    visual_data_one_by_one = True

    data_names = [
        f'{data_root}/sz.300628.csv',
        f'{data_root}/sz.300979.csv',
        # f'{data_root}/sh.600000.csv',
        # f'{data_root}/sh.600089.csv',
        # f'{data_root}/sh.601059.csv',
        # f'{data_root}/sh.603296.csv',
        # f'{data_root}/sh.603501.csv',
        # f'{data_root}/sz.000733.csv',
        # f'{data_root}/sz.001289.csv',
        # f'{data_root}/sz.002230.csv',
        # f'{data_root}/sz.002714.csv',
    ]

    if test_all_data:
        data_names = get_valid_files(data_root, from_date, to_date)
        # data_names = [os.path.join(data_root, item) for item in os.listdir(data_root)]

    # data_name = 'data_index/sh.000300.csv'
    
    # data = get_data(data_name, from_date, to_date)
    if len(data_names) == 0:
        print("No valid data files.")
        return
    
    if "hour" in data_names[0]:
        datas = [get_minutely_data(data_name, from_date, to_date) for data_name in data_names]
    elif "us" in data_names[0]:
        datas = [get_sea_data(data_name, from_date, to_date) for data_name in data_names]
    else:
        datas = [get_data(data_name, from_date, to_date) for data_name in data_names]

    stake = 1

    strategies = [
        # NewHighStrategy ,# ok
        # NewLowStrategy, # ok
        # MovingAverageStrategy, # ok
        # CombinedIndicatorStrategy, # ok
        # RSIStrategy, # ok
        # CCIStrategy,  # ok # stop loss is importance, left side trader, 
        # DoubleEmaStrategy, # ok
        # MACDTrendFollowingStrategy, # ok
        # BollingerBandsStrategy, # ok
        # RSRSStrategy, # ok
        # PriceMomumentStrategy,
        # InvertPriceMomumentStrategy,
        # PriceMomumentStrategyForUS,
        EMATrendStrategy, # good for long trend, right side trader 
        # LongLowerShadowCandlestickStrategy,
        # DiffStrategy,
        # XGBoostStrategy,
        # TurtleTradingStrategy,
    ]

    if visual_data_one_by_one:
        random.shuffle(datas)
        for data in datas:
            print(f"Processing {data._name} ...")
            test_backtrader([data], strategies=strategies, cash=cash, commission=0.001, stake=stake, visual_data=True)
    else:
        test_backtrader(datas, strategies=strategies, cash=cash, commission=0.001, stake=stake)


def demo_of_multiple_data():
    data_root = 'data'
    stake = 100
    from_date = datetime.datetime(2020, 2, 1)
    to_date = datetime.datetime(2024, 5, 31)

    total_gain = 0.0
    total_count = 0
    success_count = 0
    for item in os.listdir(data_root):
        file = os.path.join(data_root, item)
        print(f"Processing {file} ...")
        data = get_data(file, from_date, to_date)
        # data = get_minutely_data(file, from_date, to_date)
        try:

            # gain = test_backtrader(data, strategy=MovingAverageStrategy, cash=100000.0, commission=0.001, stake=stake) # -525427
            # gain = test_backtrader(data, strategy=CombinedIndicatorStrategy, cash=100000.0, commission=0.001, stake=stake) #3645
            # rsi sell-->buy：483488, buy-->sell:-242343.07.  -283477.20
            # gain = test_backtrader(data, strategy=RSIStrategy, cash=100000.0, commission=0.001, stake=stake) 

            # -359091.58,
            # gain = test_backtrader(data, strategy=CCIStrategy, cash=100000.0, commission=0.001, stake=stake)

            # gain = test_backtrader(data, strategy=DoubleEmaStrategy, cash=100000.0, commission=0.001, stake=stake)

            # gain = test_backtrader(data, strategy=NewHighStrategy, cash=100000.0, commission=0.001, stake=stake)

            # gain = test_backtrader(data, strategy=MACDTrendFollowingStrategy, cash=100000.0, commission=0.001, stake=stake)

            # gain = test_backtrader(data, strategy=BollingerBandsStrategy, cash=100000.0, commission=0.001, stake=stake)

            gain = test_backtrader(data, strategy=RSRSStrategy, cash=100000.0, commission=0.001, stake=stake)
            total_gain += gain
            total_count += 1
            if gain >= 0:
                success_count += 1
        except Exception as e:
            print(f"Error in processing {file}, {e}")
    print(f"Total gain: {total_gain}")
    print(f"{success_count} / {total_count} acc: {success_count / total_count}")


def demo_of_multiple_stock():
    pass
    data_root = 'data'
    from_date = datetime.datetime(2020, 1, 1)
    to_date = datetime.datetime(2024, 12, 31)
    cash = 100000.0
    commission = 0.001
    stake = 100


    stock_data_list = []
    cerebro = bt.Cerebro()
    cerebro.addstrategy(CCIStrategy)

    for item in os.listdir(data_root)[:20]:
        file = os.path.join(data_root, item)
        print(f"Processing {file} ...")
        data = get_data(file, from_date, to_date)
        # stock_data_list.append(data)
        cerebro.adddata(data, name=item)
    
    # Set our desired cash start
    cerebro.broker.setcash(cash)

    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.FixedSize, stake=stake)

    # Set the commission
    cerebro.broker.setcommission(commission=commission)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    results = cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Print the profit
    profit = cerebro.broker.getvalue() - cash



if __name__ == '__main__':
    demo_of_simple_strategy()
    # demo_of_multiple_data()
    # demo_of_multiple_stock()
    # demo_of_ShortTermReversalEffectinStocks()
