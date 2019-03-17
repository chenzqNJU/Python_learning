import tushare as ts
import numpy as np
import pandas as pd
import os
import gc
import myfunc
from WindPy import w
import time
from datetime import datetime, timedelta
# 初始化接口#
import json

w.start()

###########################################
################################## 交易日历

cal = w.tdays("2015-01-20").Data[0]  # 返回t1到t2的交易日历 days='Alldays''Trading''Weekdays' period='DWMQSY'
cal = [x.strftime('%Y%m%d') for x in cal]


# w.tdayscount("2013-05-01","2013-06-08")   #返回期间的交易日天数
# w.tdaysoffset(-1,t)  #返回t时刻偏移的交易日

# 日期偏移
def offset(n, day, days='Trading'):
    if not isinstance(day, str): day = day.strftime('%Y%m%d')
    day = w.tdaysoffset(n, day, days=days).Data[0][0].strftime('%Y%m%d')
    return day


# 股票池 ： sector可以 全部A股 上证A股 深证A股 沪深300 上证50   等
def wset(day, universe='全部A股', suspend=1):
    t1 = w.wset("SectorConstituent", "date=" + day + ";sector=" + universe + ";field=wind_code").Data[0]
    if suspend == 1:
        t2 = w.wset('tradesuspend', 'startdate=' + day + ';enddate=' + day + ';field=wind_code').Data[0]
        t1 = list(set(t1) - set(t2))
        t1.sort()  # 删除停牌的
    return t1


# wind返回格式 转为 dataframe 格式
def to_df(data):
    date = [x.strftime('%Y%m%d') for x in data.Times]
    if len(data.Times) > 1:
        data = pd.DataFrame(data.Data, index=data.Codes, columns=date)
    else:
        data = pd.DataFrame(data.Data, index=data.Times, columns=data.Codes).T
    return data


def wsd(symbol, attribute, time, option='', time_range=1, style='s'):
    # 一般指标比较少，取股票数、日期数较大的为index，其中，sat依次为键值、columns、index
    # if time_range > 0:begin=time;end=offset(time_range,time)
    # if time_range < 0:end = time;begin = offset(time_range, time)
    # if time_range==0:end=begin=time
    t = cal.index(time)
    tradedate = cal[min(t, t + time_range):max(t, t + time_range) + 1]
    if isinstance(attribute, str): attribute = attribute.split(',')
    if isinstance(symbol, str): symbol = symbol.split(',')

    data1 = dict()
    var = attribute[0]
    df = pd.DataFrame()
    for var in attribute:
        data = to_df(w.wsd(symbol, var, tradedate[0], tradedate[-1], option))
        data['attr'] = var
        df = df.append(data)
    df = df.reset_index()
    df.columns[1]
    if style == 't':
        for i in range(len(tradedate)):
            data1[tradedate[i]] = df.iloc[:, [0, i + 1, -1]].pivot(index='index', columns='attr',
                                                                   values=df.columns[i + 1])
    if style == 'a':
        for attr in attribute:
            data1[attr] = df[df.attr == attr].iloc[:, :-1].set_index('index')
    if style == 's':
        for s in symbol:
            data1[s] = df[df['index'] == s].iloc[:, 1:].set_index('attr').T
    if len(data1) == 1: data1 = data1[list(data1)[0]]
    return data1


# 对 股票池 进行常规性的筛检
def zt(stk, day):
    # 排除 day 日股票涨停的现象，确保当日收盘可以买入
    t = w.wsd(stk, "pre_close,close", day, day)
    t = to_df(t)
    stk = t[t.closePrice != np.round(t.preClosePrice * 1.1 + 0.0001, 2)].secID.tolist()
    return stk


def xd(stk, day):
    # 排除 day 日股票下跌的股票，确保股票止住势头
    t = DataAPI.MktEqudGet(secID=stk, tradeDate=day, field=u"secID,preClosePrice,closePrice", isOpen=1, pandas="1")
    stk = t[t.closePrice > np.round(t.preClosePrice * 0.99, 2)].secID.tolist()
    return stk


def sl(stk, day):
    # 缩量：选择成交量地量的股票
    t = DataAPI.MktEqudGet(secID=stk, tradeDate=day, field=u"secID,turnoverValue,turnoverRate", isOpen=1, pandas="1")
    stk = t[t.closePrice > np.round(t.preClosePrice * 0.99, 2)].secID.tolist()
    return stk


# 计算所选股票未来几天的收益
def ret(stk, begin, length=2, flag=0):
    end = offset(length - 1, begin)
    if flag == 0: return to_df(w.wsd(stk, "pct_chg", begin, end))
    if flag == 1:  # 后面length天相对于begin天的涨跌
        t = to_df(w.wsd(stk, "close", begin, end))
        return t.apply(lambda x: x / x[0] * 100 - 100, axis=1)
    if flag == 2:  # 后面length天最高价相对于begin天的涨跌
        t1 = to_df(w.wsd(stk, "high", begin, end))
        t2 = to_df(w.wsd(stk, "close", begin, begin))
        t = t2.join(t1.iloc[:, 1:])
        return t.apply(lambda x: x / x[0] * 100 - 100, axis=1)
    if flag == 3:  # 后面length天相当于begin第二天开盘价的涨跌
        t2 = to_df(w.wsd(stk, "close", begin, end))
        begin = offset(1, begin)
        t1 = to_df(w.wsd(stk, "open", begin, begin))
        t = np.round(t2.div(t1.iloc[:, 0], axis=0) * 100 - 100, 2)
        t.iloc[:, 0] = -100 * (t.iloc[:, 0] / (t.iloc[:, 0] + 100))  # 第一列为开盘涨跌
        return t


stk = df1.index.tolist();
begin = day_1
t2 = w.wset('tradesuspend', 'startdate=' + day_1 + ';enddate=' + end + ';field=wind_code').Data[0]

###############################################回撤函数
capital_base = 1000000
freq = 'd'
refresh_rate = 5
tradeDate =
begin = tradeDate[0]
end = tradeDate[-1]


class account:
    def __init__(self):
        self.tradeDate = begin
        self.secpos = pd.Series()  # 当前持仓股票index，及持仓量values
        self.Valid_secpos = pd.Series()
        self.price = pd.Series()  # 当前持仓股票index，及当前价格price
        self.cash = capital_base  # 现金价值
        self.Cost = pd.Series()
        self.SecValue = 0  # 股票价值
        self.PortfolioValue = capital_base  # 总价值
        self.days_counter = 0  # 天数

    def pre_tradeDate(self):
        if self.days_counter > 0:
            return (tradeDate[self.days_counter - 1])
        else:
            return (np.nan)
        # return(self.tradeDate-timedelta(1))

    def valid_secpos(self):
        self.Valid_secpos = self.secpos[self.secpos > 0]

    def referencePrice(self):  # 持仓股票的价格
        self.price = Price.loc[self.tradeDate, self.secpos.index]
        return (self.price)

    def referenceSecValue(self):  # 持仓证券的价值
        self.referencePrice()
        self.SecValue = (self.secpos * self.price).sum()
        return (self.SecValue)

    def referencePortfolioValue(self):
        self.referenceSecValue()
        self.PortfolioValue = self.cash + self.SecValue
        return self.PortfolioValue

    def order_to(self, stk, stk_num):  # 对单个股票，交易到num数量
        if stk in self.secpos.index:
            temp = stk_num - self.secpos[stk]
            self.secpos[stk] = stk_num
        else:
            temp = stk_num
            self.secpos = self.secpos.append(pd.Series({stk: stk_num}))
        self.cash -= temp * Price.loc[self.tradeDate, stk]
        self.referencePortfolioValue()

    def order(self, *args, cost=0):  # 1 对单个股票，交易num数量（'000001.XSHE ',100）
        if isinstance(args[0], str):  # 单个股票
            args = pd.Series(10 if len(args) == 1 else args[1], index=[args[0]])
        self.secpos = self.secpos.append(args)
        self.secpos = self.secpos.groupby(self.secpos.index).sum()
        self.cash -= args * Price.loc[self.tradeDate, stk]
        self.referencePortfolioValue()


###############################################
###############################################
# 提取指标，主要解决 不能多股票 多指标问题
# option ="KDJ_N=9;KDJ_IO=3"
# w.wsd('000001.SZ,000002.SZ', "kdj", day_2, day_1, "KDJ_N=9;KDJ_IO=3", period='D')

# attribute='kdj,close'
# option ="KDJ_N=9;KDJ_IO=3"
# time=day_1;time_range=1
# symbol=stk
# a=wsd(symbol, attribute[1], time, option, time_range=-2, style='a')
#
#
# data = w.wsd(symbol, attribute, begin,end,option)
# if suspend == 1:
#     t2 = w.wset('tradesuspend', 'startdate=' + day + ';enddate=' + day + ';field=wind_code').Data[0]
#     t1 = list(set(t1) - set(t2))
#     t1.sort()  # 删除停牌的
# return t1


universe = '沪深300'

# 设置 观测日期，股票池, day 为观测点今天，应该是以 day_1 昨天收盘时进行买入，观测 day 日的收益
n = -3
day = offset(n, datetime.today().strftime('%Y%m%d'))
day_1 = offset(-1, day)
day_2 = offset(-2, day)
day_3 = offset(-3, day)

uni = wset(day_1)

stk = uni[:]
data = w.wsd(stk, "kdj", day_2, day_1, "KDJ_N=9;KDJ_IO=3", period='D')
# 代码可以借鉴matlab 的w.menu()代码，完全没差别
data = to_df(data)

data1 = data[(data.iloc[:, 1] > 0) & (data.iloc[:, 0] < 0)]
stk1 = data1.index.tolist()

data1 = data[data.iloc[:, 1] > data.iloc[:, 0]]
stk1 = data1.index.tolist()

ret(stk1, day)

data = w.wsd(stk, "maxupordown", day_2, day_1)

# 找出大涨的
r = ret(uni, day)
stk_dz = r[r.iloc[:, 0] > 9.5].index.tolist()
df = wsd(stk, 'volume', day_3, time_range=-30, style='a')

#############################找出连续两个涨停的
n = -1
day = offset(n, datetime.today().strftime('%Y%m%d'))
day_1 = offset(-1, day)
day_2 = offset(-2, day)
day_3 = offset(-3, day)
stk = wset(day_1)
DAY = day_2
df = wsd(stk, 'maxupordown', DAY, time_range=-2, style='a')
df = df[~df.isnull().any(axis=1)]
df = df.astype(int)
# 找出存在缺失值的行
# df[df.isnull().any(axis=1)]
# df[df.isnull().values==True]
df1 = df[(df.iloc[:, 0] == 0) & (df.iloc[:, 1] == 1) & (df.iloc[:, 2] == 1)]

ret(df1.index.tolist(), DAY, length=3, flag=3)

df.loc['000693.SZ', :]
df.index
#############################找出盘中涨停 尾盘未涨停的
n = -1
day = offset(n, datetime.today().strftime('%Y%m%d'))
day_1 = offset(-1, day)
day_2 = offset(-2, day)
day_3 = offset(-3, day)
stk = wset(day_1)
df = wsd(stk, 'close,pre_close,high,low', day_3, option, time_range=0, style='t')

df['zt'] = np.round(df.pre_close * 1.1 + 0.0001, 2)
df1 = df[(df.zt == df.high) & (df.zt > df.close)]
####################盘中跌停
df['dt'] = np.round(df.pre_close * 0.9 + 0.0001, 2)
df1 = df[(df.dt == df.low) & (df.dt < df.close)]

######################成交量地量
df = wsd(stk, 'volume', day_3, time_range=-30, style='a')
m = df.min(axis=1)
df1 = m[m == df.iloc[:, -1]]

ret(df1.index.tolist(), day_3, 3)

ret(df1.index.tolist(), day_3, 3, flag=2)

#############################k线图
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
import datetime
from dateutil.parser import parse
import talib as ta
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)

quotes = wsd('000001.SZ', 'open,close,high,low,volume', day_3, time_range=-300, style='s')
len(quotes)
sec = '000001.SZ'
plot_k(quotes)


def plot_k(quotes, sec='stock'):
    # plt.close('all')
    __color_balck__ = '#000000'
    __color_green__ = '#00FFFF'
    __color_purple__ = '#9900CC'

    fig = plt.figure(figsize=(10, 8))
    fig.set_tight_layout(True)

    ax1 = fig.add_axes([0.05, 0.4, 0.9, 0.55])
    ax1.set_title('title')
    ax1.set_axisbelow(True)
    ax2 = fig.add_axes([0.05, 0.05, 0.9, 0.35], axis_bgcolor='w')
    ax2.set_axisbelow(True)
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_xlim(-1, len(quotes) + 1)
    ax2.set_xlim(-1, len(quotes) + 1)
    for i in range(len(quotes)):
        close_price = quotes.ix[i, 'close']
        open_price = quotes.ix[i, 'open']
        high_price = quotes.ix[i, 'high']
        low_price = quotes.ix[i, 'low']
        vol = quotes.ix[i, 'volume']
        trade_date = quotes.index[i]
        if close_price > open_price:
            ax1.add_patch(
                patches.Rectangle((i - 0.2, open_price), 0.4, close_price - open_price, fill=False, color='r'))
            ax1.plot([i, i], [low_price, open_price], 'r')
            ax1.plot([i, i], [close_price, high_price], 'r')
            ax2.add_patch(patches.Rectangle((i - 0.2, 0), 0.4, vol, fill=False, color='r'))
        else:
            ax1.add_patch(patches.Rectangle((i - 0.2, open_price), 0.4, close_price - open_price, color='g'))
            ax1.plot([i, i], [low_price, high_price], color='g')
            ax2.add_patch(patches.Rectangle((i - 0.2, 0), 0.4, vol, color='g'))
    ax1.set_title(sec, fontproperties=font, fontsize=15, loc='left', color='r')
    # ax2.set_title(u'', fontproperties=font, fontsize=15, loc='left', color='r')
    n = 10 if len(quotes) < 100 else int(len(quotes) / 10)
    ax1.set_xticks(range(0, len(quotes), n))
    ax2.set_xticks(range(0, len(quotes), n))
    # s1 = ax1.set_xticklabels([quotes.index[x] for x in ax1.get_xticks()])
    s1 = ax2.set_xticklabels([quotes.index[x] for x in ax2.get_xticks()])

    # ma5 = pd.rolling_mean(np.array(quotes['close'], dtype=float), window=5, min_periods=0)
    # ma10 = pd.rolling_mean(np.array(quotes['close'], dtype=float), window=10, min_periods=0)
    # ma20 = pd.rolling_mean(np.array(quotes['close'], dtype=float), window=20, min_periods=0)
    ma5 = quotes.close.rolling(window=5, min_periods=0).mean().values
    ma10 = quotes.close.rolling(window=10, min_periods=0).mean().values
    ma20 = quotes.close.rolling(window=20, min_periods=0).mean().values

    ax1.plot(ma5, color='b')
    ax1.plot(ma10, color='y')
    ax1.plot(ma20, color=__color_purple__)

    ax1.annotate('MA5-', xy=(len(quotes) - 30, ax1.get_yticks()[-1]), color='b', fontsize=10)
    ax1.annotate('MA10-', xy=(len(quotes) - 20, ax1.get_yticks()[-1]), color='y', fontsize=10)
    ax1.annotate('MA20-', xy=(len(quotes) - 8, ax1.get_yticks()[-1]), color=__color_purple__, fontsize=10)
    # vol5 = pd.rolling_mean(np.array(quotes['volume'], dtype=float), window=5, min_periods=0)
    # vol10 = pd.rolling_mean(np.array(quotes['volume'], dtype=float), window=10, min_periods=0)
    vol5 = quotes.volume.rolling(window=5, min_periods=0).mean().values
    vol10 = quotes.volume.rolling(window=10, min_periods=0).mean().values
    ax2.plot(vol5, color='b')
    ax2.plot(vol10, color='y')
    plt.show()
    return fig
