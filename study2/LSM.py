import numpy as np
from time import time
import pandas as pd
import math
import datetime
import tushare as ts
from arch import arch_model
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
from urllib.request import urlretrieve
import myfunc
import os
myfunc.search('pytorch')
# 下载光大银行 2017.4.5之前的股价走势图


# import tushare as ts
# df = ts.lpr_data() #取当前年份的数据
# #df = ts.lpr_data(2014) #取2014年的数据
# df.sort('date', ascending=False).head(10)

# 导入转债数据，缺失值填充
file = pd.read_excel(r'e:\bond\2018年上市转债数据.xlsx')
file = file[:-2]
file.iloc[:,8].fillna(70,inplace=True)

###### 无风险利率
shibor = pd.read_csv(r'e:\bond\Shibor1Y.csv',encoding='GBK')

#############
num = 0
stock_no = file.iloc[num,2][:6]
stock_no = str(1-int(stock_no[0]=='6'))+stock_no
date = file.iloc[num,4].date()
date_6m = date  + datetime.timedelta(days=180)
date=str(date);date_6m=str(date_6m)

r=shibor.iloc[np.where(shibor.tradeDate==date)[0],5].values[0]/100  # 无风险利率

# ################# 0:光大银行  1：厦门国贸
# num=1
# stock_no = '0' + ['601818', '600755'][num]
# date = ['2017-04-04', '2016-01-19'][num]  # 发行日期
# date_6m = ['2017-09-18', '2016-07-15'][num]  # 6月后，可以赎回日期
# # stock_no = '0' + '113011'

start_date = '20150101'
# end_date = '20190119'
end_date = '20190310'
url = 'http://quotes.money.163.com/service/chddata.html?code=' + stock_no + '&start=' + start_date + '&end=' + end_date + '&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP'

filename = 'e:\\bond\\'+stock_no + '.csv'
if not os.path.exists(filename):urlretrieve(url, filename)
stock = pd.read_csv(filename, encoding='gbk')
stock = stock.iloc[:, [0, 3]].set_index('日期').iloc[:, 0]
stock = stock.sort_index()
# 剔除停牌的日期
stock = stock[stock > 0]
# 简单收益率
stock_pct = stock.pct_change().dropna()
# 对数收益率
stock_log = stock.apply(lambda x: math.log(x)).diff().dropna()

###############导入转债数据
filename = r'e:\bond\{}.csv'.format(['光大转债', '国贸转债'][num])
bond = pd.read_csv(filename, encoding='gbk')
bond = bond[['tradeDate', 'closePrice']].set_index('tradeDate').iloc[:, 0]

############### 日期
stock_log_ = stock_log[:date]

###########循环日期，得到波动率
datelist = stock_log[date:].index
V_1m = stock_log.rolling(window=20).std().dropna() * np.sqrt(250)
V_1y = stock_log.rolling(window=250).std().dropna() * np.sqrt(250)
garch11 = arch_model(stock_log, p=1, q=1)
res = garch11.fit(update_freq=10)
V_garch = res.conditional_volatility * np.sqrt(250)
# 起始日期到6m间的天数
N_6m = len(stock[date:date_6m])

######################## 计算历史波动率，假定1month20个交易日，1year250个交易日,仅仅是4.4号的
# V_1m = stock_log_[-20:].std() * np.sqrt(250)
# V_1y = stock_log_[-250:].std() * np.sqrt(250)
# # 计算garch(1,1)波动率
# garch11 = arch_model(stock_log_, p=1, q=1)
# # res = garch11.fit()
# res = garch11.fit(update_freq=10)
# # print(res.summary())
# V_garch = res.conditional_volatility[-1] * np.sqrt(250)

######################## 计算的一些初始值
S0 = stock[:date][-1]  # 股票或指数初始的价格;
T = 6  # 期权的到期年限(距离到期日时间间隔)
# r = [0.037604, 0.037506][num]  # 无风险利率
sigma = [V_garch, V_1m, V_1y][1][date]  # 波动率(收益标准差)
M = 250 * T  # 一年考虑250步，一天一步，number of time steps
dt = T / M  # time enterval
I = 2000  # number of simulation

########################### LSM参数
N0 = 100  # 面值
# Nt = [105, 108][num]  # 到期归还本金
Nt=file.iloc[num,9]
# Cp = [4.36, 9.03][num]  # 转股价格
Cp=file.iloc[num,5]
Cr = N0 / Cp  # 转股比例
# RedeemTigger = Cp * 1.3  # 赎回触发价格
RedeemTigger=file.iloc[num,7]*Cp/100
Redeem = Cp * 1.0113  # 赎回价格
# PutTigger = Cp * 0.7  # 回售触发价格
PutTigger=file.iloc[num,8]*Cp/100

Put = Cp * 1.0113  # 回售价格
PutValue = Put * Cr  # 触发回售时，回售价值

### 最小现金流
discountVet = Nt * np.exp(-r * dt * np.array(range(M + 1)))
discountVet = discountVet[::-1]

#######################################################################
# 20000条模拟路径，每条路径 50 个时间步数
def path(step=M, price=S0):
    S = np.zeros((step + 1, I))
    S[0] = price
    np.random.seed(20000)
    start = time()
    for t in range(1, step + 1):
        z = np.random.standard_normal(I)
        S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
    end = time()
    return S
    # 估值结果
    # print('total time is %.6f seconds' % (end - start))

################################## 前20条模拟路径
# import matplotlib.pyplot as plt
# # matplotlib inline
# plt.figure(figsize=(10, 7))
# plt.grid(True)
# plt.xlabel('Time step')
# plt.ylabel('price')
# for i in range(20):
#     plt.plot(S.T[i])
# plt.show()

########################### 作图：时间序列
def pic(series,model='LSM',leg=[]):
    plt.close('all')
    plt.figure(figsize=(14, 7))
    from datetime import datetime
    x = [datetime.strptime(t, "%Y-%m-%d") for t in series.index]
    plt.grid(True)
    if isinstance(series, pd.Series):
        plt.plot(x, series)
    else:
        t=[]
        for i in range(len(series.columns)):
            exec("l%s, = plt.plot(x, series.iloc[:, %s])" %(i,i))
            exec("t.append(l%s)"%i)
        if leg==[]:plt.legend(handles=t, labels=[model + '计算价格走势',u'可转债价格走势', u'股价*转换比例走势'], loc='best', prop=font)
        else:plt.legend(handles=t, labels=leg, loc='best', prop=font)
    plt.show()

pic(res.conditional_volatility)
pic(stock_log_.rolling(window=20).std())
pic(stock)
pic(bond)

########### 判断是否赎回
def Judge_redeem(n):  # 判断是否赎回，约定若当天高于1.3，且最近30交易日有15个高于1.3
    t = S[n - 29:n + 1]
    ### 赎回情况，当日大于赎回价格
    # t1 = np.where(t[-1] >= RedeemTigger)[0]
    ### 赎回情况，最近30天、
    t2 = np.where(np.sum(t >= RedeemTigger, axis=0) >= 15)[0]
    # return np.array(list(set(t1) & set(t2)))
    return t2
########### 判断是否回售
def Judge_put(n):  # 判断是否回售，约定若当天低于0.7，且最近30交易日有15个低于1.3
    t = S[n - 29:n + 1]
    ### 赎回情况，当日大于赎回价格
    # t1 = np.where(t[-1] <= PutTigger)[0]
    ### 赎回情况，最近30天、
    t2 = np.where(np.sum(t <= PutTigger, axis=0) >= 15)[0]
    # return np.array(list(set(t1) & set(t2)))
    return t2
########### LSM计算4.4日的开售定价
def LSM():
    n = M
    # 构造现金流矩阵
    CashFlows = np.zeros((M + 1, I))
    CashFlows[n] = [max(x * Cr, Nt) for x in S[-1]]
    V = CashFlows[n]

    # 构造策略矩阵  0：无 1：到期还本 2：自愿转股 3：被迫转股 4：转债回售
    Strategy = np.zeros((M + 1, I))
    Strategy[n] = [int(x * Cr > Nt) + 1 for x in S[-1]]

    R2 = np.zeros((M + 1))
    while n > 125:
        n -= 1
        if n % 300 == 0: print(n)
        price = S[n]
        ### 赎回情况，大于赎回价格，则必须转股
        list1 = Judge_redeem(n)
        ### 回售情况，当回售价值小于贴现值则不回售
        list2 = []
        judge_put = Judge_put(n)
        if PutValue < discountVet[n]:
            list2 = judge_put
        ### 未触发回售，但是转股价值低于贴现值，则不转股，不回售
        list3 = np.array(list(set(np.where(price * Cr < discountVet[n])[0]) & (set(range(I)) - set(judge_put))))

        ### 需要回归的
        flag = np.array(list((set(range(I)) - set(list1) - set(list2) - set(list3))))

        X = price[flag].reshape(-1, 1) * Cr  # 内在价值
        Y = (V[flag] * np.exp(-r * dt)).reshape(-1, 1)  # 存续价值

        t = price.copy()
        t[judge_put] = Put  # 内在价值
        V_in = t[flag].reshape(-1, 1) * Cr

        model = LinearRegression()
        model.fit(X, Y)
        R2[n] = model.score(X, Y)
        # a = model.intercept_  # 截距
        # b = model.coef_  # 回归系数
        # print("最佳拟合线:截距", a, ",回归系数：", b)
        Y_pred = model.predict(X)

        ### 需要转股，内在价值大于预估存续价值，和强制赎回的
        flag1 = np.append(flag[np.where(V_in > Y_pred)[0]], list1).astype(int)
        if len(flag1) > 0:
            CashFlows[n, flag1] = price[flag1] * Cr
            CashFlows[n + 1:, flag1] = 0
            ### 修正策略
            Strategy[n, flag1] = 2
            Strategy[n + 1:, flag1] = 0
            if len(list1) > 0: Strategy[n, list1] = 3
            t = list(set(flag1) & set(judge_put))
            if len(t) > 0:
                CashFlows[n, t] = PutValue
                Strategy[n, t] = 4

        ### 计算n时刻的现值
        V = np.dot(np.exp(-r * dt * np.array(range(M + 1 - n))), CashFlows[n:])

    V = np.dot(np.exp(-r * dt * (1 + np.array(range(M + 1 - n)))), CashFlows[n:])
    global V_LSM
    V_LSM = V.mean() * np.exp(-r * dt * 125)
    print(V_LSM)

############# 对策略作图，R2
a1 = (Strategy == 1).sum(axis=1).cumsum()
a2 = (Strategy == 2).sum(axis=1).cumsum()
a3 = (Strategy == 3).sum(axis=1).cumsum()
a4 = (Strategy == 4).sum(axis=1).cumsum()
plt.close('all')
plt.figure(figsize=(14, 7))
plt.grid(True)
plt.plot(R2[R2 > 0])
plt.show()

# S[-1].mean()
# b=CashFlows.sum(axis=0)>0
# b.sum()
# a = CashFlows[:, 1]
# t = np.where(a > 0)[0]
# a[t]
# S[t,0] * Cr
# Cr
# Strategy[t,0]
# Strategy.sum(axis=0)

S = path()
LSM()

###########################  BSM model
# 构造现金流矩阵
CashFlows = np.zeros((M + 1, I))
CashFlows[-1] = [max(x * Cr, Nt) for x in S[-1]]
V = CashFlows[-1]
V_BSM = V.mean() * np.exp(-r * dt * 1500)

######################### 考虑赎回 回售
t = (S > RedeemTigger) + 0
t = t - (S < PutTigger)
t1 = t.cumsum(axis=0)
t2 = np.roll(t1, 30, axis=0)
t2[:30] = 0
t = t1 - t2

a, b = np.where((t == 15) | (t == -15))
import pandas as pd
c = pd.DataFrame()
c['a'] = a;
c['b'] = b
c = c.sort_values(['b', 'a'])
d = c.groupby('b').first()
for y in d.index:
    x = d.loc[y][0]
    if t[x, y] < 0:
        CashFlows[x, y] = PutValue
    else:
        CashFlows[x, y] = S[x, y] * Cr
    CashFlows[-1, y] = 0
V = np.dot(np.exp(-r * dt * np.array(range(M + 1))), CashFlows)
V_BSM_ = V.mean()


print('光大转债')
print('LSM:', V_LSM, V_LSM / 102 - 1)
print('BSM:', V_BSM, V_BSM / 102 - 1)
print('BSM_考虑:', V_BSM_, V_BSM_ / 102 - 1)

print('国贸转债')
print('LSM:', V_LSM, V_LSM / 10.3 - 1)
print('BSM:', V_BSM, V_BSM / 10.3 - 1)
print('BSM_考虑:', V_BSM_, V_BSM_ / 10.3 - 1)

##########################################################
#####################循环计算定价
def LSM_change():
    n = M
    # 构造现金流矩阵
    CashFlows = np.zeros((M + 1, I))
    CashFlows[n] = [max(x * Cr, Nt) for x in S[-1]]
    V = CashFlows[n]

    while n > max(N_6m - dateN,1):
        n -= 1
        if n % 300 == 0: print(n)
        price = S[n]
        ### 赎回情况，大于赎回价格，则必须转股
        list1 = Judge_redeem(n)
        ### 回售情况，当回售价值小于贴现值则不回售
        list2 = []
        judge_put = Judge_put(n)
        if PutValue < discountVet[n]:
            list2 = judge_put
        ### 未触发回售，但是转股价值低于贴现值，则不转股，不回售
        list3 = np.array(list(set(np.where(price * Cr < discountVet[n])[0]) & (set(range(I)) - set(judge_put))))

        ### 需要回归的
        flag = np.array(list((set(range(I)) - set(list1) - set(list2) - set(list3))))
        if len(flag)==0:continue
        X = price[flag].reshape(-1, 1) * Cr  # 内在价值
        Y = (V[flag] * np.exp(-r * dt)).reshape(-1, 1)  # 存续价值
        t = price.copy()
        t[judge_put] = Put  # 内在价值
        V_in = t[flag].reshape(-1, 1) * Cr
        model = LinearRegression()
        model.fit(X, Y)
        Y_pred = model.predict(X)

        ### 需要转股，内在价值大于预估存续价值，和强制赎回的
        flag1 = np.append(flag[np.where(V_in > Y_pred)[0]], list1).astype(int)
        if len(flag1) > 0:
            CashFlows[n, flag1] = price[flag1] * Cr
            CashFlows[n + 1:, flag1] = 0
            t = list(set(flag1) & set(judge_put))
            if len(t) > 0:
                CashFlows[n, t] = PutValue
        ### 计算n时刻的现值
        V = np.dot(np.exp(-r * dt * np.array(range(M + 1 - n))), CashFlows[n:])

    V = np.dot(np.exp(-r * dt *  np.array(range(M + 1))), CashFlows).mean()
    return V

##################### LSM
dateN = 0
P = []
for dateN in range(len(datelist)):
    # if dateN<304:continue
    date = datelist[dateN]
    print(dateN)
    # datelist[N_6m]
    sigma = V_1m[date]
    M = 1500 - 1 - dateN
    S0 = stock[date]
    S = path(M, S0)
    # S.shape
    t = LSM_change()
    P = P + [t]
########################  BSM model
# 构造现金流矩阵
dateN = 0
V_BSM = []
for dateN in range(len(datelist)):
    date = datelist[dateN]
    print(dateN)
    # datelist[N_6m]
    sigma = V_1m[date]
    M = 1500 - 1 - dateN
    S0 = stock[date]
    S = path(M, S0)
    # S.shape
    CashFlows = np.zeros((M + 1, I))
    CashFlows[-1] = [max(x * Cr, Nt) for x in S[-1]]
    V = CashFlows[-1]
    t = V.mean() * np.exp(-r * dt * M)
    V_BSM = V_BSM + [t]
######################### BSM model 考虑赎回 回售
dateN = 0
V_BSM_change = []
for dateN in range(len(datelist)):
    date = datelist[dateN]
    print(dateN)
    sigma = V_1m[date]
    M = 1500 - 1 - dateN
    S0 = stock[date]
    S = path(M, S0)
    t = (S > RedeemTigger) + 0
    t = t - (S < PutTigger)
    t1 = t.cumsum(axis=0)
    t2 = np.roll(t1, 30, axis=0)
    t2[:30] = 0
    t = t1 - t2

    a, b = np.where((t == 15) | (t == -15))
    c = pd.DataFrame()
    c['a'] = a;
    c['b'] = b
    c = c.sort_values(['b', 'a'])
    d = c.groupby('b').first()
    CashFlows = np.zeros((M + 1, I))
    CashFlows[-1] = [max(x * Cr, Nt) for x in S[-1]]
    for y in d.index:
        x = d.loc[y][0]
        if t[x, y] < 0:
            CashFlows[x, y] = PutValue
        else:
            CashFlows[x, y] = S[x, y] * Cr
        CashFlows[-1, y] = 0
    V = np.dot(np.exp(-r * dt * np.array(range(M + 1))), CashFlows)
    V_BSM_change = V_BSM_change + [V.mean()]

#############################################
###################保存数据
np.save("e:/save/path1.npy", path)
np.save("e:/save/path2.npy", path)
np.save("e:/save/p%s.npy"%str(num+1),P)
np.save("e:/save/V_BSM%s.npy"%str(num+1),V_BSM)
np.save("e:/save/V_BSM_change%s.npy"%str(num+1),V_BSM_change)

#############################################
bond_P = np.load('e:/save/p%s.npy'%str(num+1))
V_BSM = np.load('e:/save/V_BSM%s.npy'%str(num+1))
V_BSM_change = np.load('e:/save/V_BSM_change%s.npy'%str(num+1))


############# 作图 ，计算误差等
bond_P = pd.Series(bond_P,index=datelist).to_frame().join(bond).join(stock*Cr)
bond_BSM = pd.Series(V_BSM,index=datelist).to_frame().join(bond).join(stock*Cr)
bond_BSM_change = pd.Series(V_BSM_change,index=datelist).to_frame().join(bond).join(stock*Cr)

pic(bond_P)
pic(bond_BSM,model='BSM')
pic(bond_BSM_change,model='BSM改进')

bond_P['error1'] = np.abs((bond_P.iloc[:,0] - bond_P.iloc[:,1])/bond_P.iloc[:,1] *100)
bond_BSM['error2'] =  np.abs((bond_BSM.iloc[:,0] - bond_BSM.iloc[:,1])/bond_BSM.iloc[:,1] *100)
bond_BSM_change['error3'] =  np.abs((bond_BSM_change.iloc[:,0] - bond_BSM_change.iloc[:,1])/bond_BSM_change.iloc[:,1] *100)
# error = bond_P['error1'].to_frame().join(bond_BSM['error2']).join(bond_BSM_change['error3'])
error = pd.concat([bond_BSM_change['error3'],bond_P['error1'],bond_BSM['error2']],axis=1)
pic(error,leg=['LSM模型误差走势','BSM模型误差走势','BSM改进模型误差走势'])
t=error.mean()
t.index=['LSM模型误差走势','BSM模型误差走势','BSM改进模型误差走势']
print(t)

# np.convolve([1,2,3,4,5,6,7],[1/3,1/3,1/3], 'valid')
# def rolling_window(a, window):
#     shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
#     strides = a.strides + (a.strides[-1],)
#     return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
#
#
# import talib as tb
# tb.SUM(t,timeperiod=10)
# help(tb.SUM)


# GARCH(1,1) Model in Python
#   uses maximum likelihood method to estimate (omega,alpha,beta)
# (c) 2014 QuantAtRisk, by Pawel Lachowicz; tested with Python 3.5 only

# import datetime as dt
# import pandas_datareader.data as web
#
# djia = web.get_data_fred('DJIA')
# returns = 100 * djia['DJIA'].pct_change().dropna()
# from arch.univariate import arch_model
# am = arch_model(returns)
# am = arch_model(returns, mean='AR', lags=2, vol='harch', p=[1, 5, 22])
# am = arch_model(returns, mean='zero', p=1, o=1, q=1,power = 1.0, dist = 'StudentsT')
#
# r = np.array([0.945532630498276,
#               0.614772790142383,
#               0.834417758890680,
#               0.862344782601800,
#               0.555858715401929,
#               0.641058419842652,
#               0.720118656981704,
#               0.643948007732270,
#               0.138790608092353,
#               0.279264178231250,
#               0.993836948076485,
#               0.531967023876420,
#               0.964455754192395,
#               0.873171802181126,
#               0.937828816793698])
# r = returns

# from arch import arch_model
# import sys

# sys.modules['arch']

# len(r)
#
# help(arch_model)
# garch11 = arch_model(r, p=1, q=1)
# res = garch11.fit()
# res = garch11.fit(update_freq=10)
# print(res.summary())
