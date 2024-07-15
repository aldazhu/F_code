import backtrader as bt
import datetime

import os

from backtrader_strategy import *

def test_backtrader(data, strategy, cash=100000.0, commission=0.001,stake=100):
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(strategy)
    # cerebro.optstrategy(strategy, rsi_period=range(10, 31))

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

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

    return profit


def get_data(data_name, from_date, to_date):
    if not os.path.exists(data_name):
        print(f"Data file {data_name} does not exist.")
        raise "Data file does not exist."

    data = MyData(dataname=data_name,
                fromdate=from_date,
                todate=to_date)
    return data

def demo_of_simple_strategy():
    # Create a Data Feed
    # data_name = 'data/sz.300628.csv'
    data_name = 'data_index/sh.000300.csv'
    from_date = datetime.datetime(2023, 1, 1)
    to_date = datetime.datetime(2024, 12, 31)
    data = get_data(data_name, from_date, to_date)

    # test_backtrader(data, strategy=MovingAverageStrategy, cash=100000.0, commission=0.001, stake=100)

    # test_backtrader(data, strategy=QuickGuideStrategy, cash=100000.0, commission=0.001, stake=100)

    test_backtrader(data, strategy=RSIStrategy, cash=100000.0, commission=0.001, stake=1)

    # test_backtrader(data, strategy=CCIStrategy, cash=100000.0, commission=0.001, stake=1)

    # test_backtrader(data, strategy=OSCStrategy, cash=100000.0, commission=0.001, stake=100)

    # test_backtrader(data, strategy=DoubleMAStrategy, cash=100000.0, commission=0.001, stake=100)

    # test_backtrader(data, strategy=TrendFollowingStrategy, cash=100000.0, commission=0.001, stake=100)




def demo_of_multiple_stocks():
    # Create a Data Feed
    data_name = 'data/sz.300628.csv'
    from_date = datetime.datetime(2022, 11, 1)
    to_date = datetime.datetime(2024, 12, 31)
    data = get_data(data_name, from_date, to_date)

    for file in os.listdir('data'):
        data_name = f'data/{file}'
        # data_name = 'data/sh.600025.csv'
        data = get_data(data_name, from_date, to_date)
        print(f'test {data_name}')

        # test_backtrader(data, strategy=MovingAverageStrategy, cash=100000.0, commission=0.001, stake=100)

        # test_backtrader(data, strategy=QuickGuideStrategy, cash=100000.0, commission=0.001, stake=100)

        # test_backtrader(data, strategy=RSIStrategy, cash=100000.0, commission=0.001, stake=100)

        # test_backtrader(data, strategy=CCIStrategy, cash=100000.0, commission=0.001, stake=100)

        # test_backtrader(data, strategy=OSCStrategy, cash=100000.0, commission=0.001, stake=100)

        # test_backtrader(data, strategy=DoubleMAStrategy, cash=100000.0, commission=0.001, stake=100)

        # test_backtrader(data, strategy=TrendFollowingStrategy, cash=100000.0, commission=0.001, stake=100)

        test_backtrader(data, strategy=GroupStrategy, cash=100000.0, commission=0.001, stake=100)

    # test_backtrader(data, strategy=CombinedIndicatorStrategy, cash=100000.0, commission=0.001, stake=100)

def demo_of_multiple_data():
    data_root = 'data'
    from_date = datetime.datetime(2020, 1, 1)
    to_date = datetime.datetime(2024, 12, 31)

    total_gain = 0.0
    total_count = 0
    success_count = 0
    for item in os.listdir(data_root):
        file = os.path.join(data_root, item)
        print(f"Processing {file} ...")
        data = get_data(file, from_date, to_date)
        try:

            # gain = test_backtrader(data, strategy=MovingAverageStrategy, cash=100000.0, commission=0.001, stake=100) # -525427
            # gain = test_backtrader(data, strategy=CombinedIndicatorStrategy, cash=100000.0, commission=0.001, stake=100) #3645
            # rsi sell-->buy：483488, buy-->sell:-242343.07.  -283477.20
            # gain = test_backtrader(data, strategy=RSIStrategy, cash=100000.0, commission=0.001, stake=100) 

            # -359091.58,
            # gain = test_backtrader(data, strategy=CCIStrategy, cash=100000.0, commission=0.001, stake=100)

            # gain = test_backtrader(data, strategy=DoubleEmaStrategy, cash=100000.0, commission=0.001, stake=100)

            gain = test_backtrader(data, strategy=NewHighStrategy, cash=100000.0, commission=0.001, stake=100)

            # gain = test_backtrader(data, strategy=MACDTrendFollowingStrategy, cash=100000.0, commission=0.001, stake=100)

            # gain = test_backtrader(data, strategy=BollingerBandsStrategy, cash=100000.0, commission=0.001, stake=100)
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
        cerebro.adddata(data)
    
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
    # demo_of_simple_strategy()
    demo_of_multiple_data()
    # demo_of_multiple_stock()