import tushare as ts
import numpy as np
import pandas as pd
import os
import gc
os.chdir("e:\\wind")

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
# 导入需要的库
import matplotlib.pyplot as plt
import matplotlib.finance as mpf   #mpl_finance module
import mpl_finance as mpf
import talib as ta

import myfunc



# 设置历史数据区间
date1 = (2014, 12, 1)  # 起始日期，格式：(年，月，日)元组
date2 = (2016, 12, 1)  # 结束日期，格式：(年，月，日)元组
# 从雅虎财经中获取股票代码601558的历史行情
quotes = mpf.quotes_historical_yahoo_sohlc('601558.ss', date1, date2)

# 创建一个子图
fig, ax = plt.subplots(facecolor=(0.5, 0.5, 0.5))
fig.subplots_adjust(bottom=0.2)
# 设置X轴刻度为日期时间
ax.xaxis_date()
# X轴刻度文字倾斜45度
plt.xticks(rotation=45)
plt.title("股票代码：601558两年K线图")
plt.xlabel("时间")
plt.ylabel("股价（元）")
mpf.candlestick_ohlc(ax, quotes, width=1.2, colorup='r', colordown='green')
plt.grid(True)

##############################################
from matplotlib.pylab import date2num
import datetime

data=ts.get_hist_data('000005').iloc[:,:5]

#####调换顺序
t = data.close
data = data.drop('close',axis=1)
data.insert(3,'close',t)


#t=data.set_index(['high','low'])
#t=t.reset_index()
#data['dates']=data.index
#data=data.reset_index(drop=True)

# 对tushare获取到的数据转换成candlestick_ohlc()方法可读取的格式
data.dates[1]
data_list = []
for dates, row in data.iterrows():
    # 将时间转换为数字
    date_time = datetime.datetime.strptime(dates, '%Y-%m-%d')
    t = date2num(date_time)
    open, high, low, close = row[:4]
    datas = (t, open, high, low, close)
    data_list.append(datas)

# 创建子图
plt.close('all')
fig, ax = plt.subplots(figsize=(12, 6))
fig.subplots_adjust(bottom=0.2)
# 设置X轴刻度为日期时间
ax.xaxis_date()
plt.xticks(rotation=45)
plt.yticks()
plt.title("股票代码：601558两年K线图",fontproperties=font)
plt.xlabel("时间",fontproperties=font)
plt.ylabel("股价（元）",fontproperties=font)
mpf.candlestick_ohlc(ax, data_list[:100], width=0.5, colorup='r', colordown='green')
plt.grid()
plt.show()

############################################################
import requests
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

fig = plt.figure(figsize=(8, 6), dpi=72, facecolor="white")
axes = plt.subplot(111)
axes.set_title('Shangzheng')
axes.set_xlabel('time')
line, = axes.plot([], [], linewidth=1.5, linestyle='-')
alldata = []

code='sh000001'
def dapan(code):
    url = 'http://hq.sinajs.cn/?list=' + code
    r = requests.get(url)
    data = r.content[21:-3].decode('gbk').split(',')
    alldata.append(data[3])
    axes.set_ylim(float(data[5]), float(data[4]))
    return alldata
def init():
    line.set_data([], [])
    return line
def animate(i):
    axes.set_xlim(0, i + 10)
    x = range(i + 1)
    y = dapan('sh000001')
    line.set_data(x, y)
    return line

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=10000, interval=5000)
plt.show()

import myfunc
myfunc.search('sys.')
##############################
ts.get_hist_data('600848',start='2018-01-05',end='2018-01-09')

from pandas import DataFrame, Series
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib import ticker as mticker
from matplotlib.finance import candlestick_ohlc
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, YEARLY
from matplotlib.dates import MonthLocator, MONTHLY
import datetime as dt
import pylab

daylinefilespath = 'G:\\dayline\\'
stock_b_code = '000001'  # 平安银行
MA1 = 10
MA2 = 50
startdate = dt.date(2017, 6, 29)
enddate = dt.date(2018, 1, 30)
'852'.zfill(6)
sday=startdate
eday=enddate
stockcode=stock_b_code
def readstkData(rootpath, stockcode, sday, eday):
    returndata = pd.DataFrame()
    for yearnum in range(0, int((eday - sday).days / 365.25) + 1):
        theyear = sday + dt.timedelta(days=yearnum * 365)
        # build file name
        filename = rootpath + theyear.strftime('%Y') + '\\' + str(stockcode).zfill(6) + '.csv'
        try:
            rawdata = pd.read_csv(filename, parse_dates=True, index_col=0, encoding='gbk')
        except IOError:
            raise Exception('IoError when reading dayline data file: ' + filename)
        returndata = pd.concat([rawdata, returndata])
    # Wash data
    returndata = returndata.sort_index()
    returndata.index.name = 'DateTime'
    returndata.drop('amount', axis=1, inplace=True)
    returndata.columns = ['Open', 'High', 'Close', 'Low', 'Volume']

    returndata = returndata[returndata.index < eday.strftime('%Y-%m-%d')]

    return returndata



data=ts.get_hist_data('000005').iloc[:,:5]
data=data.reset_index()
data.date=pd.to_datetime(data.date)
daysreshape=data
daysreshape['date'][1]
daysreshape['date'].astype(dt.date)[1]

MA2=10;MA1=5

plt.close('all')
def main():
    #利用tushare导入数据
    days = ts.get_hist_data('000005',start='2017-11-01',end='2018-04-26').iloc[:,:5]
    # drop the date index from the dateframe & make a copy
    daysreshape = days.reset_index()
    daysreshape.date = pd.to_datetime(daysreshape.date)
    # convert the datetime64 column in the dataframe to 'float days'
    daysreshape['DateTime'] = mdates.date2num(daysreshape['date'].astype(dt.date))
    # clean day data for candle view
    daysreshape.drop(['date'], axis=1, inplace=True)
    #调整顺序
    daysreshape = daysreshape.reindex(columns=['DateTime', 'open', 'high', 'low', 'close'])

    Av1 = movingaverage(daysreshape.close.values, MA1)
    Av2 = movingaverage(daysreshape.close.values, MA2)
    #SP = len(daysreshape.DateTime.values[MA2 - 1:])  #有效期间，剔除前9天的数据
    SP=50
    fig = plt.figure(facecolor='#07000d', figsize=(15, 10))
    ax1 = plt.subplot2grid((6, 4), (1, 0), rowspan=4, colspan=4, axisbg='#07000d')
    #蜡烛图
    candlestick_ohlc(ax1, daysreshape.values[-SP:], width=1, colorup='#ff1717', colordown='#53c156')
    Label1 = str(MA1) + ' SMA'
    Label2 = str(MA2) + ' SMA'

    ax1.plot(daysreshape.DateTime.values[-SP:], Av1[-SP:], '#e1edf9', label=Label1, linewidth=1.5)
    ax1.plot(daysreshape.DateTime.values[-SP:], Av2[-SP:], '#4ee6fd', label=Label2, linewidth=1.5)
    ax1.grid(True, color='w')
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.yaxis.label.set_color("w")
    #上下左右的边框线为蓝色
    ax1.spines['bottom'].set_color("#5998ff")
    ax1.spines['top'].set_color("#5998ff")
    ax1.spines['left'].set_color("#5998ff")
    ax1.spines['right'].set_color("#5998ff")
    ax1.tick_params(axis='y', colors='w')
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    ax1.tick_params(axis='x', colors='w')
    plt.ylabel('Stock price and Volume')

    ######################### plot an RSI indicator on top
    maLeg = plt.legend(loc=9, ncol=2, prop={'size': 7},
                       fancybox=True, borderaxespad=0.)
    maLeg.get_frame().set_alpha(0.4)
    textEd = pylab.gca().get_legend().get_texts()
    pylab.setp(textEd[0:5], color='w')
    ####subplot2grid对面板分区，（6,4）总面板，（0,0）当前所在位置，rowspan高度为1 colspan长度为4
    ax0 = plt.subplot2grid((6, 4), (0, 0), sharex=ax1, rowspan=1, colspan=4, axisbg='#07000d')
    rsi = ta.RSI(daysreshape.close.values,timeperiod=10)

    rsiCol = '#c1f9f7'
    posCol = '#386d13'
    negCol = '#8f2020'

    ax0.plot(daysreshape.DateTime.values[-SP:], rsi[-SP:], rsiCol, linewidth=1.5)
    ax0.axhline(70, color=negCol)  #划线
    ax0.axhline(30, color=posCol)
    ax0.fill_between(daysreshape.DateTime.values[-SP:], rsi[-SP:], 70, where=(rsi[-SP:] >= 70), facecolor=negCol,
                     edgecolor=negCol, alpha=0.5)
    ax0.fill_between(daysreshape.DateTime.values[-SP:], rsi[-SP:], 30, where=(rsi[-SP:] <= 30), facecolor=posCol,
                     edgecolor=posCol, alpha=0.5)
    ax0.set_yticks([30, 70])
    ax0.yaxis.label.set_color("w")

    ax0.spines['bottom'].set_color("#5998ff")
    ax0.spines['top'].set_color("#5998ff")
    ax0.spines['left'].set_color("#5998ff")
    ax0.spines['right'].set_color("#5998ff")
    ax0.tick_params(axis='y', colors='w')
    ax0.tick_params(axis='x', colors='w')
    plt.ylabel('RSI')

    volumeMin = 0
    ax1v = ax1.twinx()
    #填充颜色
    ax1v.fill_between(daysreshape.DateTime.values[-SP:], volumeMin, days.volume.values[-SP:], facecolor='#00ffe8', alpha=.4)
    ax1v.axes.yaxis.set_ticklabels([])
    ax1v.grid(False)
    ###Edit this to 3, so it's a bit larger
    ax1v.set_ylim(0, 3 * days.volume.values.max())
    ax1v.spines['bottom'].set_color("#5998ff")
    ax1v.spines['top'].set_color("#5998ff")
    ax1v.spines['left'].set_color("#5998ff")
    ax1v.spines['right'].set_color("#5998ff")
    ax1v.tick_params(axis='x', colors='w')
    ax1v.tick_params(axis='y', colors='w')

    # plot an MACD indicator on bottom
    ax2 = plt.subplot2grid((6, 4), (5, 0), sharex=ax1, rowspan=1, colspan=4, axisbg='#07000d')
    fillcolor = '#00ffe8'
    nslow = 26
    nfast = 12
    nema = 9
    #macd 是快线减慢线，ema是对差的移动平均，macdhist是macd与其移动平均的差
    #emaslow, emafast, macd = computeMACD(daysreshape.Close.values)
    macd, macdsignal, macdhist = ta.MACD(daysreshape.close.values, fastperiod=12, slowperiod=26, signalperiod=9)

    #ema9 = ExpMovingAverage(macd, nema)  计算指数移动平均，即macdsignal
    ax2.plot(daysreshape.DateTime.values[-SP:], macd[-SP:], color='#4ee6fd', lw=2)
    ax2.plot(daysreshape.DateTime.values[-SP:], macdsignal[-SP:], color='#e1edf9', lw=1)
    ax2.fill_between(daysreshape.DateTime.values[-SP:], macdhist[-SP:], 0, alpha=0.5, facecolor=fillcolor,
                     edgecolor=fillcolor)
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    ax2.spines['bottom'].set_color("#5998ff")
    ax2.spines['top'].set_color("#5998ff")
    ax2.spines['left'].set_color("#5998ff")
    ax2.spines['right'].set_color("#5998ff")
    ax2.tick_params(axis='x', colors='w')
    ax2.tick_params(axis='y', colors='w')
    plt.ylabel('MACD', color='w')
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='both'))
    #标签转45度
    for label in ax2.xaxis.get_ticklabels():
        label.set_rotation(45)

    ### 添加标题 删除上面的x轴日期
    plt.suptitle(stock_b_code, color='w')
    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax1.get_xticklabels(), visible=False)


    # 用箭头标记 第30天的 移动平均线，分别箭头位置、文字位置等
    ax1.annotate('BreakNews!', (daysreshape.DateTime.values[-30], Av1[-30]),
                 xytext=(0.8, 0.9), textcoords='axes fraction',
                 arrowprops=dict(facecolor='white', shrink=0.05),
                 fontsize=10, color='w',
                 horizontalalignment='left', verticalalignment='bottom')
    #对子图进行微调，修减做图框外的空白
    plt.subplots_adjust(left=.09, bottom=.14, right=.94, top=.95, wspace=.20, hspace=0)

    plt.show()

if __name__ == "__main__":
    main()

help(mticker.MaxNLocator)

###########中心移动平均， # yy(1) = y(1) yy(2) = (y(1) + y(2) + y(3))/3 yy(3) = (y(1) + y(2) + y(3) + y(4) + y(5))/5
a=np.arange(10)
window_size=5
#首尾的除数仍然是5，convolve计算数组的 卷 积
window = np.ones(int(window_size))/float(window_size)
np.convolve(a, window, 'same')
#和卷积函数convolve一样，没有首尾
cumsum_vec = np.cumsum(np.insert(a, 0, 0))
ma_vec = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size

out0 = np.convolve(a, np.ones(window_size, dtype=int), 'valid') / window_size
#out0 = np.convolve(a, np.ones(window_size, dtype=int)/window_size, 'valid')
r = np.arange(1, window_size - 1, 2)
start = np.cumsum(a[:window_size - 1])[::2] / r
stop = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
np.concatenate((start, out0, stop))

window_size=4
def movingaverage(data, window_size):
    t1 = np.convolve(data, np.ones(window_size, dtype=int)/window_size, 'valid')
    t2 = np.cumsum(data[:window_size-1])/np.arange(1,window_size)
    return np.concatenate((t2, t1))

###############计算macd值
def get_EMA(df,N):
    for i in range(len(df)):
        if i==0:
            df.ix[i,'ema']=df.ix[i,'close']
        if i>0:
            df.ix[i,'ema']=(2*df.ix[i,'close']+(N-1)*df.ix[i-1,'ema'])/(N+1)
    ema=list(df['ema'])
    return ema
def get_MACD(df,short=12,long=26,M=9):
    a=get_EMA(df,short)
    b=get_EMA(df,long)
    df['diff']=pd.Series(a)-pd.Series(b)
    #print(df['diff'])
    for i in range(len(df)):
        if i==0:
            df.ix[i,'dea']=df.ix[i,'diff']
        if i>0:
            df.ix[i,'dea']=(2*df.ix[i,'diff']+(M-1)*df.ix[i-1,'dea'])/(M+1)
    df['macd']=df['diff']-df['dea']
    return df
#diff  dee macd 分别是快线 慢线 柱体
a=daysreshape.loc[:,['close']]
get_MACD(a,12,26,9)
#等价于
macd, macdsignal, macdhist = ta.MACD(daysreshape.close.values, fastperiod=12, slowperiod=26, signalperiod=9)

with open(str(ta.__file__),"r") as f:
    print (f.read())