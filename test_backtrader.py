import backtrader as bt
import datetime

import os

from backtrader_strategy import *

def test_backtrader(data, strategy, cash=100000.0, commission=0.001,stake=100):
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(strategy)

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
    cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    cerebro.plot()


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


if __name__ == '__main__':
    demo_of_simple_strategy()
    # demo_of_multiple_stocks()