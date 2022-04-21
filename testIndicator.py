import dataPro as dp
import indicator
import numpy as np
import matplotlib.pyplot as plt

def testCCI(data:"pandas frame", days:int) -> None :
    CCI = indicator.CCI(data,days)
    len_CCI = len(CCI)
    print(f"len(CCI) = {len_CCI}")
    print("CCI[days:days+days]:",CCI[len_CCI - days: len_CCI])

def testCCIWithPrice(data,days:int, Ndays=25) -> None:
    """
    看CCI和之后N天的价格变化关系散点图
    """
    CCI = indicator.CCI(data, days)
    len_CCI = len(CCI)
    close = data['close']
    chgPercent = [(close[i] - close[i-Ndays])/close[i-Ndays]  for i in range(days + Ndays, len_CCI)]
    x = np.array(CCI[days:len_CCI-Ndays])#CCI从第days天开始计算，所以起点是dsys
    y = np.array(chgPercent)*100#改变量是从有CCI开始的后面第Ndays天计算，所以起点是days + Ndays

    fig, axs = plt.subplots(2,2)
    axs[0,0].set_title("CCI-chg")
    axs[0,0].set_xlabel("CCI")
    axs[0,0].set_ylabel("chg")
    axs[0,0].scatter(x, y)

    axs[0, 1].set_title("date-CCI")
    axs[0, 1].set_ylabel("CCI")
    axs[0, 1].plot(np.arange(len(x)),x)

    axs[1, 1].set_title("date- after Ndays chg")
    axs[1, 1].set_ylabel("chg")
    axs[1, 1].set_xlabel("date")
    axs[1, 1].plot(np.arange(len(y)),y)

    plt.show()



def testMA(data:"pandas frame", days:int) -> None:
    MA = indicator.MA(data,days)
    len_MA = len(MA)
    print(f"len_MA = {len_MA}")
    print(f"pre {days} close price: {data[len_MA-days:len_MA]}")
    print(f"pre {days} MA : {MA[len_MA-days:len_MA]}")


if __name__ == "__main__":
    filePath = "data\sh.600703.csv"
    days = 15
    data = dp.readData(filePath)
    #testCCI(data,days)
    testCCIWithPrice(data,days)
    #testMA(data,days)