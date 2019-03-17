import numpy as np
import pandas as pd
import os
import gc
os.chdir("e:\\stock")   #修改当前工作目录
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
from matplotlib import rc
rc('mathtext', default='regular')
import seaborn as sns
sns.set_style('white')

import decimal
from decimal import Decimal
context=decimal.getcontext() # 获取decimal现在的上下文
context.rounding = decimal.ROUND_05UP    #'ROUND_HALF_EVEN'
round(2.55,1)
round(Decimal(2.55), 1) # 2.6
format(Decimal(2.55), '.1f') #'2.6'


factor_data = pd.read_csv('RelativePE_FullA.csv').iloc[-50:,:10]
factor_data.tradeDate=pd.to_datetime(factor_data.tradeDate, format='%Y-%m-%d')
factor_data[factor_data.index>tradeDate]

factor_data.set_index('tradeDate',inplace=True)
tradeDate='2018-02-02'
tradeDate=datetime.strptime(tradeDate,'%Y-%m-%d')
t = pd.Timestamp('2018-02-02 00:00:00')
factor_data.loc[t]
factor_data.loc[tradeDate]
t=factor_data.index[1]
t.timetuple()
time.mktime(t.timetuple())
t.time()

p=pd.read_csv('Price.csv')
for i in p.columns:
    if i.startswith('300104'):
        print(i)
a=p.columns
b=p[a[a.str.startswith('601997')]]
b1=isOpen1['300104.XSHE']
#停盘了依然有price
########################################################
########################################################
############################导入
factor_data = pd.read_csv('RelativePE_FullA.csv').iloc[-50:,:10]
factor_data.tradeDate=pd.to_datetime(factor_data.tradeDate, format='%Y-%m-%d')
factor_data.set_index('tradeDate',inplace=True)

Price = pd.read_csv('Price.csv').iloc[-50:,:10]     # 读取因子数据
Price.tradeDate=pd.to_datetime(Price.tradeDate, format='%Y-%m-%d')
Price.set_index('tradeDate',inplace=True)

############################导入isopen
isOpen = pd.read_csv('isOpen.csv').iloc[-100:,:10]
isOpen.tradeDate=pd.to_datetime(isOpen.tradeDate, format='%Y-%m-%d')
isOpen.set_index('tradeDate',inplace=True)
############################导入:最高价 最低价 开盘价

for i in ['openPrice','lowestPrice','highestPrice']:
    locals()[i] = pd.read_csv(i + '.csv').iloc[-100:, :10]
    locals()[i].tradeDate = pd.to_datetime(locals()[i].tradeDate, format='%Y-%m-%d')
    locals()[i].set_index('tradeDate', inplace=True)
    locals()[i] = locals()[i].loc[Price.index]

RSV=(Price-lowestPrice)/(highestPrice-lowestPrice)*100
RSV.replace(np.inf, np.nan,inplace=True)
#涨跌停
temp=Price[:1].copy();temp[:]=np.nan
temp1=round((Price+0.000001)*1.1,2)
temp1=temp.append(temp1[:-1])
temp1.index=Price.index
zhangting=temp1==Price
temp2=round((Price+0.000001)*0.9,2)
temp2=temp.append(temp2[:-1])
temp2.index=Price.index
dieting=(temp2==Price)

RSV[dieting]=0
RSV[zhangting]=100
####计算k值
for n,i in enumerate(RSV.index):
    #n=1;i=Price.index[n]
    if n==0:
        K=np.ones((1,RSV.shape[1]))*50;
        tempK=K[0].copy()
        D=K.copy()
        tempD=tempK.copy()
        RSV_ = RSV.values
    else:
        x=np.where(isOpen.loc[i]==1)
        tempK[x]=RSV_[n,x]*1/3+tempK[x]*2/3
        #temp[np.isnan(temp)^np.isnan(Price_[n,])]=50
        K=np.row_stack((K,tempK))
        tempD[x]=tempK[x]*1/3+tempD[x]*2/3
        D=np.row_stack((D,tempD))

K=pd.DataFrame(K,index=Price.index,columns=Price.columns).round(2)
D=pd.DataFrame(D,index=Price.index,columns=Price.columns).round(2)
K_=pd.DataFrame(K.unstack(),columns=['K'])
D_=pd.DataFrame(D.unstack(),columns=['D'])
kdj=pd.concat([K_,D_], axis=1)
kdj['J']=kdj.K*2-kdj.D

import talib as ta
help(ta.TA_RSI)

indicators=pd.DataFrame()
indicators['k'],indicators['d']=ta.STOCH(highestPrice.iloc[:,1],lowestPrice.iloc[:,1],Price.iloc[:,1],
                                         fastk_period=9,slowk_period=3,slowk_matype=0,slowd_period=3,slowd_matype=0)
MA=[5,10,20]
indicators['ma1'] = Price.iloc[:,1].rolling(MA[0]).mean()
indicators['ma2'] = Price.iloc[:,1].rolling(MA[1]).mean()
indicators['ma3'] = Price.iloc[:,1].rolling(MA[2]).mean()

ta.get_functions()

indicators['closePrice'] = data['closePrice']

ta.STOCH(11,11,12)
help(ta.AROON)
dir(ta.STOCH)
ta.STOCH.__doc__
def Get_kd_ma(data):
    indicators={}
    #计算kd指标
    indicators['k'],indicators['d']=ta.STOCH(np.array(data['highPrice']),np.array(data['lowPrice']),np.array(data['closePrice']),\
    fastk_period=9,slowk_period=3,slowk_matype=0,slowd_period=3,slowd_matype=0)
    indicators['ma1']=pd.rolling_mean(data['closePrice'], MA[0])
    indicators['ma2']=pd.rolling_mean(data['closePrice'], MA[1])
    indicators['ma3']=pd.rolling_mean(data['closePrice'], MA[2])
    indicators['ma4']=pd.rolling_mean(data['closePrice'], MA[3])
    indicators['ma5']=pd.rolling_mean(data['closePrice'], MA[4])
    indicators['closePrice']=data['closePrice']
    indicators=pd.DataFrame(indicators)
    return indicators


def Get_all_indicators(hist):
    stock_pool = []
    all_data = {}
    for i in hist:
        try:
            indicators = Get_kd_ma(hist[i])
            all_data[i] = indicators
        except Exception as e:
            # print 'error:%s'%e
            pass
        if indicators.iloc[-2]['k'] < indicators.iloc[-2]['d'] and indicators.iloc[-1]['k'] > indicators.iloc[-2]['d']:
            stock_pool.append(i)
        elif indicators.iloc[-1]['k'] >= 10 and indicators.iloc[-1]['d'] <= 20 and indicators.iloc[-1]['k'] > \
                indicators.iloc[-2]['k'] and indicators.iloc[-2]['k'] < indicators.iloc[-3]['k']:
            stock_pool.append(i)
    return stock_pool, all_data


pd.stack()

b=K.unstack().unstack()
a=K.stack()
b=a.unstack(0)
c=b.unstack()
z=a.index.values
z[3,0]
z.name=['a','b']
a.index.name=['a']
a.swaplevel('b','a')
a.index[1]

index = pd.MultiIndex.from_tuples(a.index, names=['first', 'second'])
a.index=index
a.index.names=['a','b']
a1=a.swaplevel('b','a')
a.unstack('a')

print(a[:100])
a.index[1]
z[1]
z1=list(z)
z2=list(zip(*z))[0]

isOpen=isOpen.loc[Price.index]
Price.ix[:5,4]=np.nan;isOpen.ix[:5,4]=np.nan;
k_data=pd.DataFrame(np.ones((1,9))*50,columns=Price.columns) #不能排除nan
Price_=Price.values
set(Price.index)==set(isOpen.index)


mad = lambda x: x[-1]/x[0]-1
ret=Price.rolling(window=2).apply(mad)*100

"""
for n,i in enumerate(Price.index):
    if n==1:k_data=Price_[1,]*0+50;temp=k_data
    else:
        temp=Price_[n,]*1/3+temp*2/3
        temp[np.isnan(temp)^np.isnan(Price_[n,])]=50
        k_data=np.row_stack((k_data,temp))

a=np.row_stack((k_data,temp))

t1=Price.iloc[1]
t2=Price.iloc[6]
t1.isnull()^t2.isnull()
np.isnan(temp)^np.isnan(k_data)
k_data.isempty()
"""

n=2
t=Price.index[1]
Price[3:4]
np.arange(9).reshape(9,1)
#factor_data = factor_data[factor_data.columns[:]].set_index('tradeDate')
#factor_dates = factor_data.index.values
#tradeDate = pd.Timestamp('2018-02-05')

capital_base = 1000000                    # 起始资金
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测
refresh_rate = 5
tradeDate=factor_data.index
begin=tradeDate[0]
end=tradeDate[-1]

class account:
    def __init__(self):
        self.tradeDate = begin
        self.secpos=pd.Series()   # 当前持仓股票index，及持仓量values
        self.Valid_secpos=pd.Series()
        self.price=pd.Series()    # 当前持仓股票index，及当前价格price
        self.cash=capital_base    #现金价值
        self.Cost=pd.Series()
        self.SecValue=0  #股票价值
        self.PortfolioValue=capital_base #总价值
        self.days_counter=0       #天数
    def pre_tradeDate(self):
        if self.days_counter>0:return(tradeDate[self.days_counter-1])
        else:return(np.nan)
        #return(self.tradeDate-timedelta(1))
    def valid_secpos(self):
        self.Valid_secpos=self.secpos[self.secpos>0]
    def referencePrice(self):  # 持仓股票的价格
        self.price=Price.loc[self.tradeDate,self.secpos.index]
        return(self.price)

    def referenceSecValue(self):  # 持仓证券的价值
        self.referencePrice()
        self.SecValue=(self.secpos*self.price).sum()
        return(self.SecValue)
    def referencePortfolioValue(self):
        self.referenceSecValue()
        self.PortfolioValue=self.cash+self.SecValue
        return self.PortfolioValue
    def order_to(self,stk,stk_num):     #对单个股票，交易到num数量
        if stk in self.secpos.index:
            temp=stk_num-self.secpos[stk]
            self.secpos[stk]=stk_num
        else:
            temp=stk_num
            self.secpos=self.secpos.append(pd.Series({stk:stk_num}))
        self.cash-=temp*Price.loc[self.tradeDate,stk]
        self.referencePortfolioValue()
    def order(self, *args,cost=0):   # 1 对单个股票，交易num数量（'000001.XSHE ',100）
        if isinstance(args[0],str):  #单个股票
            args=pd.Series(10 if len(args) == 1 else args[1],index=[args[0]])
        self.secpos=self.secpos.append(args)
        self.secpos=self.secpos.groupby(self.secpos.index).sum()
        self.cash -= args * Price.loc[self.tradeDate, stk]
        self.referencePortfolioValue()




def z(*args):
    print(len(args))
    print(type(args[0]))
    print(10 and (len(args) - 1) and args[1])
    print(10 if len(args) == 1 else args[1])
    return pd.Series(10 if len(args) == 1 else args[1],index=['11'])
z('dfd',)
args=('fdf',)

a.pre_tradeDate()
temp=pd.Series([100,100,100],index=Price.columns[1:4])

a=account()
a.secpos=temp
a.referenceSecValue()
a.referencePortfolioValue()
a.order_to('000005.XSHE',100)
a.order('000004.XSHE',-100)
a.secpos
a.valid_secpos()
a.Valid_secpos

t[t>0]
a.referencePrice('000005.XSHE',100)
a.cash
a.price
t1=a.secpos
t=pd.Series(100,index=[Price.columns[3]])
t1=t1.append(t,inplace=True)

isinstance(t,pd.Series)
isinstance('ff',str)
temp=temp.append(t)
a=temp.groupby(temp.index).sum()
temp.index
temp.add(t,ign)

type((a.price*a.secpos).sum())
a.tradeDate
a.referenceSecValue()
a.referenceSecValue

'000005.XSHE' in t1.index
t1['000005.XSHE']
t1.append(pd.Series({'000005.XSHE':100}))
class User(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def SetName(self, name):
        self.name = name

    def SetAge(self, age):
        self.age = age

    def GetName(self):
        return self.name

    def GetAge(self):
        return self.age


u = User('kzc', 17)
u.GetName()
u.GetAge()

from functools import reduce
import functools
help(reduce)
reduce(add, [1,2,3,4]
functools.__file__
reduce.__file__

temp=pd.Series(range(10),index=Price.columns[:10])

for i in (Price.index):
    a=Price.loc[i]

    if n>49:
print(n);

seq = ['one', 'two', 'three']
for i, element in enumerate(seq):
    if 1 < 2:
        print(type(i), seq[i])

    if n==0:
        temp=Price[i]

if 1>2:
    1


#分析得到市值数据
mkt_value_data = pd.read_csv('MarketValue.csv')
mkt_value_data.set_index('tradeDate',inplace=True)
a=mkt_value_data.index
b=a.month
b_=b[1:].append('0')

b=list(a.month)
b_=b[1:]+[0]
c=np.array(b)-np.array(b_)

m=mkt_value_data.ix[c!=0,:]
m=np.log(m)


z+[1]
m1=m.stack()
len(mkt_value_data)
len(c)
m1.to_csv('mktvalue.csv',encoding='gbk')

mkt_value_data.index=mkt_value_data.tradeDate
mkt_value_data['2016-03-07':'2016-03-09']