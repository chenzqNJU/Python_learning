import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime
import tushare as ts
import os
import gc
os.chdir("e:\\stock\\temp")

import myfunc
from importlib import reload
reload(myfunc)

myfunc.search('npy')
##############股票列表
data=ts.get_stock_basics()
code=data.index
code=code.sort_values()
np.save('code.npy',code)
code = np.load('code.npy')

data=pd.DataFrame();bug=[]
process_bar = myfunc.ShowProcess(len(code))
for x in code:
    t=ts.get_hist_data(x,start='2018-03-01',end='2018-05-09')
    #t = ts.get_hist_data('601965') 为空
    if t is None:bug.append(x)
    else:
        t = ts.get_hist_data(x,start='2018-03-01',end='2018-05-09').iloc[:,:5]
        t['code']=x
        data = data.append(t)
    process_bar.show_process()
#############################################
data=pd.read_csv('month2.csv')
data=data.iloc[:,1:]
data.ticker=data.ticker.astype(str)
data.ticker=data.ticker.str.zfill(6)
#将停牌的价格为空值,删除
data.drop(list(np.where(data.turnoverVol==0)[0]),inplace=True)

Price=pd.pivot_table(data,index='tradeDate',columns='ticker',values='closePrice')

Price_=Price.fillna(method='pad')  #填充缺失值
zhangting=myfunc.zhangdie(Price)
zhangting[zhangting==-1]=0
# 找出有两个连续涨停的
mad = lambda x: x[0]*x[1]
t=zhangting.rolling(window=2).apply(mad)
t1=t.apply(lambda x: x.sum())
t1=t1[t1>=1]

Price1=Price[t1.index.values.tolist()]
zhangting1=zhangting[t1.index.values.tolist()]
#标记出连续涨停的始末
zhangting2=zhangting1.copy()
zhangting2.loc[:,:]=0
t=zhangting1.values
t=np.vstack((t,np.zeros((1,np.shape(t)[1])))).astype(int)

for x in range(len(zhangting2)):
    t1 = np.where(t[x+1,:]*t[x,:]*(1-t[x-1,:]))
    t1=t1[0].tolist()
    zhangting2.iloc[x,t1]=1
    t2 = np.where(t[x-1,:]*t[x,:]*(1-t[x+1,:]))
    t2=t2[0].tolist()
    zhangting2.iloc[x,t2]=2

t=(zhangting2==1).sum()
#涨停顶点价格
t=Price1[zhangting2==2]
t.max()
# 回撤
mad = lambda x: x[-1]
t=zhangting1.rolling(window=2).apply(mad)
#累加，只考虑3-12号之前就有连续涨停的
t=zhangting2.cumsum()
t1=t.loc['2018-03-12']>0
t2=t.ix[:,t1]
#提取出符合要求的data、Price
Price2=Price1.loc[:,t1]
data1=data[data.ticker.isin(Price2.columns)]


reload(myfunc)
df=data[data.ticker=='000555']
myfunc.kxiantu(df,code=code).show()

process_bar = myfunc.ShowProcess(len(t1))
############### 输入 data 及 code 做出k线图：Price2.columns
it=t1.index[:4]
it=Price2.columns
for code in it:
    df=data[data.ticker==code]
    z=myfunc.kxiantu(df,code=code)
    z.savefig('e:\\stock\\temp\\'+code+'.png')

z.show()

import baostock as bs
lg = bs.login(user_id="anonymous", password="123456")
rs = bs.query_history_k_data("sz.000005", "date,code,open,high,low,close",
    start_date='2017-07-01', end_date='2018-04-26',
    frequency="d", adjustflag="3")
rs.get_row_data()

############################导入沪深300股票数据
code=ts.get_hs300s()
data=pd.DataFrame();bug=[]
process_bar = myfunc.ShowProcess(len(code))
for x in code[:100].code:
    t=ts.get_hist_data(x,start='2017-10-01',end='2018-04-26')
    #t = ts.get_hist_data('601965') 为空
    if t is None:bug.append(x)
    else:
        t = ts.get_hist_data(x,start='2017-10-01',end='2018-04-26').iloc[:,:5]
        t['code']=x
        data = data.append(t)
    process_bar.show_process()

data.to_csv('t.csv')
data=pd.read_csv('t.csv',index_col=[0])

############
data=data.reset_index()
myfunc.flag_jishu(data)
data.date=pd.to_datetime(data.date)
data.set_index(['date',"flag"],inplace=True)
data1=data.sort_index(level=0)
data1=data1.unstack().stack(dropna=False)

a=data1.close[1]
data2=data1.round(2)
a=data1.close.astype(float)
type(a[1])
c=a[1]
b=a[1].astype('int')

pd.set_option('precision', 2)

data2=data1.unstack()
data2=data2.stack(dropna=False)

help(pd.DataFrame.stack)
len(data3)
data3=data3.dropna()

date=data1.index.levels[0]

myfunc.search('index.level')

data1.index=data1.index.droplevel()
data1.loc[10]



#600485一直停牌
set(code[:100].code)-set(data.code)
######################################判断是否有flag计算未考虑情况
myfunc.flag_jishu(data)
for  i in range(1,len(data)-1):
    if np.abs(data.iloc[i,3]/data.iloc[i-1,3]-1)>0.11 and data.iloc[i,6]==data.iloc[i-1,6]:
        t=t+[i]
t=t+list(np.array(t)-1)
t.sort()
data.iloc[t,:]


data_ = DataAPI.FutuGet(contractStatus="L", field=u"contractObject", pandas="1")
universe = [x + 'M0' for x in set(data_.contractObject)]
start = '2017-10-01'  # 回测开始时间
end = '2018-04-26'  # 回测结束时间
capital_base = 1e10  # 初试可用资金
refresh_rate = 2  # 调仓周期
freq = 'd'  # 调仓频率：m-> 分钟；d-> 日
commission = Commission(0.00005, 0.00005, 'perValue')

z=data.date.tolist()
[z.count(x) for x in z].min()
day=38  #先看第50天
def handle_data(futures_account):  # 回测调仓逻辑，每个调仓周期运行一次，可在此函数内实现信号生产，生成调仓指令。

    symbols = get_symbol(universe)

    prices = data1.loc[date[day]].close
    #选择为停牌的
    code=prices[~ prices.isnull()].index

    buy_long = []
    window = 30
    r = 0.15
    #symbol =data1.index.levels[1][0] #先看第一个股票
    symbol=code[0]
    symbol=7
    for symbol in code:
        #data = pd.DataFrame(get_symbol_history(symbol, time_range=window + 3)[symbol]).replace(0, np.NAN)
        #data = data1.loc[date[10 - 3:10].tolist()]

        data=data1.loc[[i for i in data1.index if i[1] == symbol and i[0] in date[day - window -2:day+1]]]
        data = data.dropna(axis=0)  # 排除未交易日
        if len(data) == 0: continue
        High = np.array(data['high'])[-window:]
        Low = np.array(data['low'])[-window:]
        Close = np.array(data['close'])[-window:]
        Volume = np.array(data['volume'])[-window:]
        #if len(Volume) < window: continue
        local_high = []
        local_low = []
        index_low = []
        index_high = []

        for i in range(2, window - 2):
            if High[i] >= max(High[i - 2:i + 3]):
                local_high.append(High[i])
                index_high.append(i)
            if Low[i] <= min(Low[i - 2:i + 3]):
                local_low.append(Low[i])
                index_low.append(i)
        local_low = local_low[-3:]
        local_high = local_high[-3:]
        if len(local_high) <= 1 or len(local_low) <= 1 or sorted(local_low) != local_low:
            continue

plt.plot(Low)
plt.show()
        #price = np.array(prices[symbol]['closePrice'])[-1]
        price = prices[symbol]

        # 突破
        if (index_low[-1] > index_high[-1] and price >= 1.02 * np.mean(local_high)
                and max(local_high) - min(local_high) < r * min(local_high)
                and Volume[-1] > 1.2 * Volume[index_low[-1]:-1].mean()):
            buy_long.append(symbol)
        # 回踩
        elif (len(local_high) > 2 and index_low[-1] < index_high[-1]
              and max(local_high[:-1]) - min(local_high[:-1]) < r * min(local_high[:-1])
              and price > 1.02 * np.mean(local_high[:-1]) and local_high[-1] > 1.02 * np.mean(local_high[:-1])
              and Volume[index_high[-1]] > 1.2 * Volume[index_low[-1]:index_high[-1]].mean()):
            buy_long.append(symbol)

    sell_list = []
    for symbol in futures_account.position.keys():
        long_position = futures_account.position[symbol]['long_position']
        try:
            price = np.array(prices[symbol]['closePrice'])[-1]
        except:  # 这里实际是处理当合约更换时，平仓
            sell_list.append(symbol)
            order(symbol, -long_position, 'close')
            continue
        if ((price <= futures_account.mod_cost[symbol] * 0.92 or price >= futures_account.mod_cost[symbol] * 1.2)
            and symbol not in buy_long) or not (price > 0):
            sell_list.append(symbol)
            order(symbol, -long_position, 'close')

    buy_long += list(set(futures_account.position.keys()).difference(set(sell_list)))
    n = float(len(buy_long))
    if n == 0: return


    per_cash = futures_account.portfolio_value / n
    for symbol in buy_long:
        coef = np.array(DataAPI.FutuGet(ticker=symbol, field=u"contMultNum", pandas="1").contMultNum)[-1]
        long_position = futures_account.position.get(symbol, dict()).get('long_position', 0)
        price = np.array(prices[symbol]['closePrice'])[-1]

        if long_position > 0:
            aux = max(futures_account.mod_cost[symbol], price)
            futures_account.mod_cost[symbol] += (aux - futures_account.mod_cost[symbol]) * 2 / 5.0
        else:
            futures_account.mod_cost[symbol] = price

        amount = int(per_cash / price / coef) - long_position
        if amount > 0:
            order(symbol, amount, 'open')
        elif amount < 0:
            order(symbol, amount, 'close')

myfunc.search('global')

date=data1.index.levels[0].tolist()
x=date[0]
begin=x
capital_base = 100000

begin=date[3]
del begin
locals().keys()
################################################################################
class account:
    def __new__(cls,*args,**kwargs):
        cls.date=date
        cls.begin=date[0] if 'begin' not in globals().keys() else begin
        cls.cash=10000 if 'capital_base' not in globals().keys() else capital_base
        return object.__new__(cls,*args,**kwargs)
    def __iter__(self):
        self.days_counter=-1
        return self
    def __next__(self):
        if self.days_counter == len(self.date) -1 :
            raise StopIteration
        else:
            self.days_counter+=1
            self.tradeDate=self.date[self.days_counter]
        return self
    def __init__(self):
        self.tradeDate = self.begin
        self.secpos=pd.Series()   # 当前持仓股票index，及持仓量values
        self.Valid_secpos=pd.Series()
        self.price=pd.Series()    # 当前持仓股票index，及当前价格price
        self.Cost=pd.Series()
        self.SecValue=pd.Series({self.begin:0}) #股票价值
        self.PortfolioValue=pd.Series({self.begin:self.cash}) #总价值
        self.days_counter=self.date.index(self.begin)       #天数从0计数
    def pre_tradeDate(self):
        if self.days_counter>0:return(date[self.days_counter-1])
        else:return(np.nan)
        #return(self.tradeDate-timedelta(1))
    def valid_secpos(self):
        self.Valid_secpos=self.secpos[self.secpos>0]
    def referencePrice(self):  # 持仓股票的价格
        self.price=Price.loc[self.tradeDate,self.secpos.index]
        return(self.price)

    def referenceSecValue(self):  # 持仓证券的价值
        self.referencePrice()
        self.SecValue[self.tradeDate]=(self.secpos*self.price).sum()
        return(self.SecValue)
    def referencePortfolioValue(self):
        self.referenceSecValue()
        self.PortfolioValue[self.tradeDate]=self.cash+self.SecValue
        return self.PortfolioValue

    def order_to(self,stk,stk_num):     #对单个股票，交易到num数量
        if stk in self.secpos.index:
            temp=stk_num-self.secpos[stk]
            self.secpos[stk]=stk_num
        else:
            temp=stk_num
            print(temp)
            self.secpos=self.secpos.append(pd.Series({stk:stk_num}))
        print(Price.loc[self.tradeDate,stk])
        self.cash-=temp*Price.loc[self.tradeDate,stk]
        self.referencePortfolioValue()
    def order(self, *args,cost=0):   # 1 对单个股票，交易num数量（'000001.XSHE ',100）
        if isinstance(args[0],str):  #单个股票
            args=pd.Series(100 if len(args) == 1 else args[1],index=[args[0]])
        else:args = args[0]
        self.secpos=self.secpos.append(args)
        self.secpos=self.secpos.groupby(self.secpos.index).sum()
        self.cash -= (args * Price.loc[self.tradeDate, args.index]).sum()
        self.referencePortfolioValue()
        if self.cash<0:print('error:cash<0')



data1=data1.dropna()
data1.code=data1.code.astype(int).astype('str')
Price=pd.pivot_table(data1,columns='code',values='close',index='date')

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


len(date)
a.order_to('600000',200)
a.__next__()

a=account();
a.days_counter=-1
for n,x in enumerate(a):
    if n==4:break
    print(a.tradeDate)
    if n<2:a.order('600008')
a.referencePortfolioValue()


x.__next__()

a=account()
for i in a:
    print(a.days_counter)

for i in range(6):
    print(a.days_counter)
    a=a.__next__()

    print(a.__next__())

class a:
    def __init__(self):
        self.tradeDate = [date[0] if 'begin' not in globals().keys() else begin]
z=a()


a.tradeDate
a.tradeDate=date[12]
a.referencePortfolioValue()

t=pd.Series([100,100,100],index=Price.columns[1:4])
t.append(pd.Series({'d':400}),inplace=True)
t['d']=45

a.order(t)
a.order('600008')

if 1:raise StopIteration



z.sum()

t.sum()
a.tradeDate
a.pre_tradeDate()
a.Valid_secpos

myfunc.search('pivot')

for x in date:

class test():
    def __init__(self,data=1):
        self.data = data
    def __iter__(self):
        return self
    def __next__(self):
        if self.data > 6:
            raise StopIteration
        else:
            self.data+=1
            return self.data

t = test()
for i in range(3):
    print(t.__next__())
for item in t:
    print(type(item))
    print(item)


import xml.etree.ElementTree as ET
tree = ET.parse('S41.xml')

root=tree.getroot()
root.attrib
z=root[400][0]

#遍历子节点
child=root[1]
for child in root:
    t=[]
    for i in range(4):
        t+=[child[i].text]

try:
    tree = ET.ElementTree(file='S41.xml')
except Exception as e:
    # Do some log operation
    tree = None
