import mplfinance as mpf
import matplotlib as mpl# 用于设置曲线参数
from cycler import cycler# 用于定制线条颜色
import pandas as pd# 导入DataFrame数据
import matplotlib.pyplot as plt
import numpy as np


def plotCandle(data,code):   
    # 设置基本参数
    # type:绘制图形的类型，有candle, renko, ohlc, line等
    # 此处选择candle,即K线图
    # mav(moving average):均线类型,此处设置7,30,60日线
    # volume:布尔类型，设置是否显示成交量，默认False
    # title:设置标题
    # y_label:设置纵轴主标题
    # y_label_lower:设置成交量图一栏的标题
    # figratio:设置图形纵横比
    # figscale:设置图形尺寸(数值越大图像质量越高)
    kwargs = dict(
        type='candle', 
        mav=(7, 30,60), 
        volume=True, 
        title='\nA_stock %s candle_line' % (code),    
        ylabel='OHLC Candles', 
        ylabel_lower='Shares\nTraded Volume', 
        figratio=(15, 10), 
        figscale=10)
    
    # 设置marketcolors
    # up:设置K线线柱颜色，up意为收盘价大于等于开盘价
    # down:与up相反，这样设置与国内K线颜色标准相符
    # edge:K线线柱边缘颜色(i代表继承自up和down的颜色)，下同。详见官方文档)
    # wick:灯芯(上下影线)颜色
    # volume:成交量直方图的颜色
    # inherit:是否继承，选填
    mc = mpf.make_marketcolors(
        up='red', 
        down='green', 
        edge='i', 
        wick='i', 
        volume='in', 
        inherit=True)

    # 设置图形风格
    # gridaxis:设置网格线位置
    # gridstyle:设置网格线线型
    # y_on_right:设置y轴位置是否在右
    s = mpf.make_mpf_style(
        gridaxis='both', 
        gridstyle='-.', 
        y_on_right=False, 
        marketcolors=mc)

    # 设置均线颜色，配色表可见下图
    # 建议设置较深的颜色且与红色、绿色形成对比
    # 此处设置七条均线的颜色，也可应用默认设置
    mpl.rcParams['axes.prop_cycle'] = cycler(
        color=['dodgerblue', 'deeppink', 
        'navy', 'teal', 'maroon', 'darkorange', 
        'indigo'])

    # 设置线宽
    mpl.rcParams['lines.linewidth'] = .5
    
    # 图形绘制
    # show_nontrading:是否显示非交易日，默认False
    # savefig:导出图片，填写文件名及后缀
    # mpf.plot(data,
    #     **kwargs,
    #     style=s,
    #     show_nontrading=False,
    #     savefig='A_stock-%s_candle_line'
    #      % (code) + '.jpg')
    mpf.plot(data, 
        **kwargs, 
        style=s, 
        show_nontrading=False)
    

def plotEarningRatio(ratioList,flagList,data):
    x = np.arange(len(ratioList))
    ratio = np.array(ratioList) * 100
    close = np.array(data['close'])
    open = np.array(data['open'])
    #singnal =

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('test')
    ax1.grid()

    ax1.plot(x, close, '.-',c="r",label="close")
    ax1.plot(x, open, '.-',c="g",label="open")
    ax1.legend()
    ax1.set_ylabel('close')

    ax2.grid()
    ax2.plot(x, ratio, '.-',c='g')
    ax2.set_xlabel('time (d)')
    ax2.set_ylabel('ratio')
    #plt.legend()
    plt.show()


def plotMyMomentDot(flg_l, close_p):
    for i in range(len(flg_l) - 3):
        if flg_l[i] == 1 :
            mark = "v"
            color = "r"
        elif flg_l[i] == -1:
            mark = "d"
            color = "g"
        elif flg_l[i] == 0:
            mark = "."
            color = "b"
        plt.scatter(i,close_p[i],marker=mark,c=color)
    plt.show()
    

def MA(data,days:int):
    ma = [x for x in data["close"][:days]]
    for i in range(days,len(data["close"])):
        ma.append(sum(data["close"][i-days:i])/days)
    return ma

def plotHist(datalist):
    np.array(datalist)
    plt.grid()
    plt.xlabel("chg")
    plt.ylabel("p")
    plt.hist(datalist)
    plt.show()


def covariance(data1,data2):
    #calcute the covariance of the two stock
    #cov(x,y) = E[(x - X_)*(y - y_)],x_=x.mean

    #Pearson correlation coefficient :
    #coe(x,y) =  cov(x,y)/(d(x)*d(y)),d(x)= standard deviation of x
    # x = data1['open'] - data1['close']
    # y = data2['open'] - data2['close']
    x = data1['pctChg']
    y = data2['pctChg']
    # x = pd.DataFrame(x)
    # y = pd.DataFrame(y)
    cov = x.corr(y)
    return cov
