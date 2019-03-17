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

from matplotlib import dates
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)

#import statsmodels.api as sm

import scipy.stats as st


#删除变量：包含pd np等
for i in dir():
    if not i.startswith('__'):
        locals().pop(i)
#删除最近创建的n个变量
for key in list(globals().keys())[-8:]:
    if not key.startswith("__"):  # 排除系统内建函数
        globals().pop(key)

gc.collect()

# 导入PE变量
data1=pd.read_csv('factor1.csv')
data2=pd.read_csv('factor2.csv')
data3=pd.read_csv('factor3.csv')
data=pd.concat([data1,data2,data3])
data=data.set_index(['tradeDate'])
data.shape
data.dropna(how='all',inplace=True)
#很可恶的是2016-12-31仅有一个股票有值，造成无法删除
temp=np.where(data.isnull().sum(axis=1)>3000)
data.drop(data.index[temp],inplace=True)
# 导入收盘价数据
data1=pd.read_csv('closePrice1.csv')
data2=pd.read_csv('closePrice2.csv')
data3=pd.read_csv('closePrice3.csv')
data=pd.concat([data1,data2,data3])
data=data.set_index(['tradeDate'])
data.shape
Price=data[list(filter(lambda x:x[0] in '036',data.columns))].dropna(how='all')
Price.shape
Price.to_csv('Price.csv')



## 导入市值数据
data1=pd.read_csv('marketValue1.csv')
data2=pd.read_csv('marketValue2.csv')
data3=pd.read_csv('marketValue3.csv')
data=pd.concat([data1,data2,data3])
data=data.set_index(['tradeDate'])
MV=data[list(filter(lambda x:x[0] in '036',data.columns))].dropna(how='all')
MV.shape
MV.to_csv('MarketValue.csv')


factor_df=data
# 暴力去除PE极端值
factor_df[(factor_df<0) | (factor_df>500)] = np.NaN
# 计算相对市盈率因子
factor_mean_se = factor_df.mean(axis=1)
factor_df = factor_df.apply(lambda x: x / factor_mean_se)
# window为120即求均值的6个月
factor_df = factor_df.rolling(window=120).mean() / factor_df

# 前面的数据不太完整，删掉
factor_df = factor_df[factor_df.index>='2009-09-01']
#保存
factor_df.to_csv('RelativePE_FullA.csv')

###################计算之前20天的收益
Price = pd.read_csv('Price.csv')
#该方法比较占内存
mad = lambda x: x[-1]/x[0]-1
back_20d_return_data=Price.rolling(window=6).apply(mad)

Price.set_index('tradeDate')
back_20d_return_data=Price[:-20]
t=Price[:20].copy()
t[:]=np.nan
back_20d_return_data=t.append(back_20d_return_data,ignore_index=True)
back_20d_return_data.index=Price.index
#forward_5d_return_data[:5]=np.nan
back_20d_return_data=Price/back_20d_return_data-1
back_20d_return_data.to_csv('back_20d_return_data.csv')

#################计算未来5天的收益
Price.set_index('tradeDate',inplace=True)
forward_5d_return_data=Price[5:]
temp=Price[:5].copy()
temp[:]=np.nan
forward_5d_return_data=forward_5d_return_data.append(temp,ignore_index=True)
forward_5d_return_data.index=Price.index
forward_5d_return_data=forward_5d_return_data/Price-1
forward_5d_return_data.to_csv('forward_5d_return_data.csv')

gc.collect()
#########################################################################
#########################################################################
#########################################################################
####################################导入数据#############################
mkt_value_data = pd.read_csv('MarketValue.csv',index_col=[0],parse_dates = True)[-500:]
mkt_value_data = pd.read_csv('MarketValue.csv')[-500:]
forward_5d_return_data = pd.read_csv('forward_5d_return_data.csv')[-500:]
back_20d_return_data = pd.read_csv('back_20d_return_data.csv')[-500:]
factor_data = pd.read_csv('RelativePE_FullA.csv')[-500:]

a=mkt_value_data.tradeDate[:5]
b=mkt_value_data[mkt_value_data.tradeDate.year()>2008]
b=year(a)


def string_toDatetime(string):
    return datetime.strptime(string, "%Y-%m-%d")
type(string_toDatetime('2009-09-01'))
factor_data['tradeDate'] = list(map(string_toDatetime,factor_data['tradeDate']))
factor_data = factor_data[factor_data.columns[:]].set_index('tradeDate')

# 市值数据
mkt_value_data['tradeDate'] = list(map(string_toDatetime,mkt_value_data['tradeDate']))
mkt_value_data.set_index('tradeDate',inplace=True)
#未来5天
forward_5d_return_data['Unnamed: 0'] = list(map(string_toDatetime,forward_5d_return_data['Unnamed: 0']))
forward_5d_return_data.set_index('Unnamed: 0',inplace=True)
#过去20天
back_20d_return_data['Unnamed: 0'] = list(map(string_toDatetime,back_20d_return_data['Unnamed: 0']))
back_20d_return_data.set_index('Unnamed: 0',inplace=True)

gc.collect()
# 因子历史表现

n_quantile = 10
# 统计十分位数
cols_mean = ['meanQ' + str(i + 1) for i in range(n_quantile)]
cols = cols_mean
corr_means = pd.DataFrame(index=factor_data.index, columns=cols)

dt=corr_means.index[1]
# 计算相关系数分组平均值
for dt in corr_means.index:
    qt_mean_results = []

    # 相关系数去掉nan
    tmp_factor = factor_data.ix[dt].dropna()

    pct_quantiles = 1.0 / n_quantile
    for i in range(n_quantile):
        down = tmp_factor.quantile(pct_quantiles * i)
        up = tmp_factor.quantile(pct_quantiles * (i + 1))
        mean_tmp = tmp_factor[(tmp_factor <= up) & (tmp_factor >= down)].mean()
        qt_mean_results.append(mean_tmp)
    corr_means.ix[dt] = qt_mean_results

# ------------- 因子历史表现作图 ------------------------


fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(111)
x=corr_means.index.copy().to_datetime()
x=[x_.date() for x_ in x]
lns1 = ax1.plot(x, corr_means.meanQ1, label='Q1')
lns2 = ax1.plot(x, corr_means.meanQ5, label='Q5')
lns3 = ax1.plot(x, corr_means.meanQ10, label='Q10')

lns = lns1 + lns2 + lns3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, bbox_to_anchor=[0.5, 0.1],loc='best', ncol=3, mode="", borderaxespad=0., fontsize=12)
ax1.set_ylabel(u'因子', fontproperties=font, fontsize=16)
ax1.set_xlabel(u'日期', fontproperties=font, fontsize=16)
ax1.set_title(u"因子历史表现", fontproperties=font, fontsize=16)
plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(dates.YearLocator())
ax1.grid()
plt.show()

plt.close('all')

# 计算了每一天的**因子**和**之后5日收益**的秩相关系数
forward_5d_return_data.index[1]

ic_data = pd.DataFrame(index=factor_data.index, columns=['IC', 'pValue'])

# 计算相关系数

for dt in ic_data.index:
    if dt not in forward_5d_return_data.index:
        continue

    tmp_factor = factor_data.ix[dt]
    tmp_ret = forward_5d_return_data.ix[dt]
    fct = pd.DataFrame(tmp_factor)
    ret = pd.DataFrame(tmp_ret)
    fct.columns = ['fct']
    ret.columns = ['ret']
    fct['ret'] = ret['ret']
    fct = fct[~np.isnan(fct['fct'])][~np.isnan(fct['ret'])]
    #fct.dropna(how='any')
    if len(fct) < 5:
        continue

    ic, p_value = st.spearmanr(fct['fct'], fct['ret'])  # 计算秩相关系数 RankIC
    ic_data['IC'][dt] = ic
    ic_data['pValue'][dt] = p_value

ic_data.dropna(inplace=True)

print('mean of IC: ', ic_data['IC'].mean()),
print('median of IC: ', ic_data['IC'].median())
print('the number of IC(all, plus, minus): ', (len(ic_data), len(ic_data[ic_data.IC > 0]), len(ic_data[ic_data.IC < 0])))

# 每一天的**因子**和**之后5日收益**的秩相关系数作图
ic_data.drop('pValue',1,inplace=True)
plt.close('all')
fig = plt.figure(figsize=(16, 6))
ax1 = fig.add_subplot(111)

ic=ic_data.IC
x=ic_data.index.copy()
x=np.array([x_.date() for x_ in x])

temp=ic_data.IC.rolling(window=21).mean()
temp_x=[x_.date() for x_ in ic_data.index]
lns1 = ax1.plot(x[(np.where(ic > 0)[0])], ic[ic > 0] , '.r', label='IC(plus)')
lns2 = ax1.plot(x[(np.where(ic < 0)[0])], ic[ic < 0], '.b', label='IC(minus)')
lns3 = ax1.plot(temp_x,temp, 'green', label='IC(month mean)')

lns = lns1 + lns2 + lns3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, bbox_to_anchor=[0.6, 0.1], loc='best', ncol=3, mode="", borderaxespad=0., fontsize=12)
ax1.set_ylabel(u'相关系数', fontproperties=font, fontsize=16)
ax1.set_xlabel(u'日期', fontproperties=font, fontsize=16)
ax1.set_title(u"因子和之后5日收益的秩相关系数", fontproperties=font, fontsize=16)
ax1.set_ylim(-0.35, 0.35)
plt.gca().xaxis.set_major_locator(dates.MonthLocator())
ax1.grid()
plt.show()

#############看超额收益
n_quantile = 10
# 统计十分位数
cols_mean = [i + 1 for i in range(n_quantile)]
cols = cols_mean

excess_returns_means = pd.DataFrame(index=factor_data.index
                                    , columns=cols)

# 计算因子分组的超额收益平均值
dt=excess_returns_means.index[1]

for dt in excess_returns_means.index:
    if dt not in forward_5d_return_data.index:
        continue
    qt_mean_results = []

    tmp_factor = factor_data.ix[dt].dropna()
    tmp_return = forward_5d_return_data.ix[dt].dropna()
    tmp_return = tmp_return[tmp_return < 0.5]
    tmp_return_mean = tmp_return.mean()

    pct_quantiles = 1.0 / n_quantile

    for i in range(n_quantile):
        down = tmp_factor.quantile(pct_quantiles * i)
        up = tmp_factor.quantile(pct_quantiles * (i + 1))
        i_quantile_index = tmp_factor[(tmp_factor <= up) & (tmp_factor >= down)].index
        mean_tmp = tmp_return[i_quantile_index].mean() - tmp_return_mean
        qt_mean_results.append(mean_tmp)

    excess_returns_means.ix[dt] = qt_mean_results

excess_returns_means.dropna(inplace=True)

# 因子分组的超额收益作图
plt.close('all')
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(111)

excess_returns_means_dist = excess_returns_means.mean()
excess_dist_plus = excess_returns_means_dist[excess_returns_means_dist > 0]
excess_dist_minus = excess_returns_means_dist[excess_returns_means_dist < 0]
lns2 = ax1.bar(excess_dist_plus.index, excess_dist_plus.values, align='center', color='r', width=0.35)
lns3 = ax1.bar(excess_dist_minus.index, excess_dist_minus.values, align='center', color='g', width=0.35)

ax1.set_xlim(left=0.5, right=len(excess_returns_means_dist) + 0.5)  #设置x轴0.5-10.5
# ax1.set_ylim(-0.008, 0.008)
ax1.set_ylabel(u'超额收益', fontproperties=font, fontsize=16)
ax1.set_xlabel(u'十分位分组', fontproperties=font, fontsize=16)
ax1.set_xticks(excess_returns_means_dist.index)
ax1.set_xticklabels([int(x) for x in ax1.get_xticks()], fontproperties=font, fontsize=14)    #设置标签为整数
ax1.set_yticklabels([str(x * 100) + '0%' for x in ax1.get_yticks()], fontproperties=font, fontsize=14) #设置标签为百分数
ax1.set_title(u"因子选股分组超额收益", fontproperties=font, fontsize=16)
ax1.grid()
plt.show()


# 计算因子分组的市值分位数平均值
signal_df=factor_data
mkt_df=mkt_value_data
def quantile_mkt_values(signal_df, mkt_df):
    n_quantile = 10
    # 统计十分位数
    cols_mean = [i + 1 for i in range(n_quantile)]
    cols = cols_mean
    mkt_value_means = pd.DataFrame(index=signal_df.index, columns=cols)
    # 计算分组的市值分位数平均值
    for dt in mkt_value_means.index:
        if dt not in mkt_df.index:
            continue
        qt_mean_results = []

        tmp_factor = signal_df.ix[dt].dropna()
        tmp_mkt_value = mkt_df.ix[dt].dropna()
        tmp_mkt_value = tmp_mkt_value.rank() / len(tmp_mkt_value)
        pct_quantiles = 1.0 / n_quantile
        for i in range(n_quantile):
            down = tmp_factor.quantile(pct_quantiles * i)
            up = tmp_factor.quantile(pct_quantiles * (i + 1))
            i_quantile_index = tmp_factor[(tmp_factor <= up) & (tmp_factor >= down)].index
            mean_tmp = tmp_mkt_value[i_quantile_index].mean()
            qt_mean_results.append(mean_tmp)
        mkt_value_means.ix[dt] = qt_mean_results
    mkt_value_means.dropna(inplace=True)
    return mkt_value_means.mean()


# 计算因子分组的市值分位数平均值
origin_mkt_means = quantile_mkt_values(factor_data, mkt_value_data)

# 因子分组的市值分位数平均值作图
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(111)
width = 0.3
lns1 = ax1.bar(origin_mkt_means.index, origin_mkt_means.values, align='center', width=width)

ax1.set_ylim(0.3, 0.6)
ax1.set_xlim(left=0.5, right=len(origin_mkt_means) + 0.5)
ax1.set_ylabel(u'市值百分位数', fontproperties=font, fontsize=16)
ax1.set_xlabel(u'十分位分组', fontproperties=font, fontsize=16)
ax1.set_xticks(origin_mkt_means.index)
ax1.set_xticklabels([int(x) for x in ax1.get_xticks()], fontproperties=font, fontsize=14)
ax1.set_yticklabels([str(x * 100) + '0%' for x in ax1.get_yticks()], fontproperties=font, fontsize=14)
ax1.set_title(u"因子分组市值分布特征", fontproperties=font, fontsize=16)
ax1.grid()
plt.show()

plt.close('all')

#因子分组选股的一个月反转
n_quantile = 10
# 统计十分位数
cols_mean = [i + 1 for i in range(n_quantile)]
cols = cols_mean
hist_returns_means = pd.DataFrame(index=factor_data.index, columns=cols)

# 因子分组的一个月反转分布特征
dt=hist_returns_means.index[1];i=1;
start_time = time.time()
for dt in hist_returns_means.index:
    if dt not in back_20d_return_data.index:
        continue
    qt_mean_results = []

    # 去掉nan
    tmp_factor = factor_data.ix[dt].dropna()
    tmp_return = back_20d_return_data.ix[dt].dropna()
    tmp_return_mean = tmp_return.mean()

    pct_quantiles = 1.0 / n_quantile
    for i in range(n_quantile):
        down = tmp_factor.quantile(pct_quantiles * i)
        up = tmp_factor.quantile(pct_quantiles * (i + 1))
        i_quantile_index = tmp_factor[(tmp_factor <= up) & (tmp_factor >= down)].index
        mean_tmp = tmp_return[i_quantile_index].mean() - tmp_return_mean
        qt_mean_results.append(mean_tmp)

    hist_returns_means.ix[dt] = qt_mean_results
print(str(time.time() - start_time) + ' seconds elapsed in total.')

hist_returns_means.dropna(inplace=True)

# 一个月反转分布特征作图
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

hist_returns_means_dist = hist_returns_means.mean()
lns1 = ax1.bar(hist_returns_means_dist.index, hist_returns_means_dist.values, align='center', width=0.35)
lns2 = ax2.plot(excess_returns_means_dist.index, excess_returns_means_dist.values, 'o-r')

ax1.legend(lns1, ['20 day return(left axis)'], loc='best', fontsize=12)
ax2.legend(lns2, ['excess return(right axis)'],loc='right', fontsize=12)
ax1.set_xlim(left=0.5, right=len(hist_returns_means_dist) + 0.5)
ax1.set_ylabel(u'历史一个月收益率', fontproperties=font, fontsize=16)
ax2.set_ylabel(u'超额收益', fontproperties=font, fontsize=16)
ax1.set_xlabel(u'十分位分组', fontproperties=font, fontsize=16)
ax1.set_xticks(hist_returns_means_dist.index)
ax1.set_xticklabels([int(x) for x in ax1.get_xticks()], fontproperties=font, fontsize=14)
ax1.set_yticklabels([str(x * 100) + '%' for x in ax1.get_yticks()], fontproperties=font, fontsize=14)
ax2.set_yticklabels([str(x * 100) + '0%' for x in ax2.get_yticks()], fontproperties=font, fontsize=14)
ax1.set_title(u"因子选股一个月历史收益率（一个月反转因子）分布特征", fontproperties=font, fontsize=16)
ax1.grid()
plt.show()
















a=corr_means.index
xs = [datetime.strptime(d, '%m/%d/%Y').date() for d in dates]
b=corr_means.index[:3]
autodates = dates.AutoDateLocator()

dates.date2num(b)


#时间轴 作图  生成横纵坐标信息

fig = plt.figure(0)
dates1 = ['01/02/1991', '01/03/1991', '01/04/1991']
xs = [datetime.strptime(d, '%m/%d/%Y').date() for d in dates1]
type(xs[1])
type(a[1])
xs=dates.date2num(xs)
xs=a[:3].values
xs=[x.date() for x in z2]
z1[1].date()
ys = range(len(xs))
# 配置横坐标
plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%m/%d/%Y'))
plt.gca().xaxis.set_major_locator(dates.DayLocator())
# Plot
plt.plot(xs, ys)
plt.show()
plt.close('all')


datetime.fromtimestamp(a[1])

fig = plt.figure(0) # 新图 0
plt.savefig() # 保存
plt. close(0)

xs[1]

dtime = datetime.now()
un_time = time.mktime(dtime.timetuple())
type(un_time)

dates.date2num(a[1])

dates.num2date(int(dates.date2num(dtime)))
dates.num2date(dates.date2num(xs))

t = pd.Timestamp('2013-12-25 00:00:00')

t.date()
datetime.date(2013, 12, 25)
t.to_datetime()
t.date() == datetime.date(2013, 12, 25)

z=forward_5d_return_data.ix[:3,:3]

z.index[1]
z.index=z.index.to_datetime()
z1=z.index.copy()
z2=z1.to_datetime()
[x.date() for x in z2]
z1.values()
import time

"""
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def getUqerStockFactors(begin, end, factor, universe=None):
   
    使用优矿的因子DataAPI，拿取所需的因子，并整理成相应格式，返回的因子数据
    格式为pandas MultiIndex Series，例如：
    secID        tradeDate 
    002130.XSHE  2016-12-20    0.0640
                 2016-12-21    0.0643
                 2016-12-22    0.0631
                 2016-12-23    0.0641
                 2016-12-26    0.0667
    注意：后面拿取因子的DataAPI可以自行切换为专业版API，以获取更多因子支持

    Parameters
    ----------
    begin : str
        开始日期，'YYYY-mm-dd' 或 'YYYYmmdd'
    end : str
        截止日期，'YYYY-mm-dd' 或 'YYYYmmdd'
    universe: list
        股票池，格式为uqer股票代码secID，若为None则默认拿取全部A股
    factor: str
        因子名称，uqer的DataAPI.MktStockFactorsOneDayGet(或
        专业版对应API)可查询的因子

    Returns
    -------
    df : pd.DataFrame
        因子数据，index为tradeDate，columns为secID
 

    # 拿取上海证券交易所日历
    cal_dates = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=begin, endDate=end)
    cal_dates = cal_dates[cal_dates['isOpen'] == 1].sort('calendarDate')
    # 工作日列表
    cal_dates = cal_dates['calendarDate'].values.tolist()

    print
    factor + ' will be calculated for ' + str(len(cal_dates)) + ' days:'
    count = 0
    secs_time = 0
    start_time = time.time()

    # 按天拿取因子数据，并保存为一个dataframe
    df = pd.DataFrame()
    for dt in cal_dates:
        # 拿取数据dataapi，必要时可以使用专业版api
        dt_df = DataAPI.MktStockFactorsOneDayGet(tradeDate=dt, secID='',
                                                 field=['tradeDate', 'secID'] + [factor])
        if df.empty:
            df = dt_df
        else:
            df = df.append(dt_df)

        # 打印进度部分，每200天打印一次
        if count > 0 and count % 200 == 0:
            finish_time = time.time()
            print
            count,
            print
            '  ' + str(np.round((finish_time - start_time) - secs_time, 0)) + ' seconds elapsed.'
            secs_time = (finish_time - start_time)
        count += 1

    # 提取所需的universe对应的因子数据
    df = df.set_index(['tradeDate', 'secID'])[factor].unstack()
    if universe:
        universe = list(set(universe) & set(df.columns))
        df = df[universe]
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d')

    # 将上市不满三个月的股票的因子设置为NaN
    equ_info = DataAPI.EquGet(equTypeCD=u"A", secID=u"", ticker=u"", listStatusCD=u"", field=u"", pandas="1")
    equ_info = equ_info[['secID', 'listDate', 'delistDate']].set_index('secID')
    equ_info['delistDate'] = [x if type(x) == str else end for x in equ_info['delistDate']]
    equ_info['listDate'] = pd.to_datetime(equ_info['listDate'], format='%Y-%m-%d')
    equ_info['delistDate'] = pd.to_datetime(equ_info['delistDate'], format='%Y-%m-%d')
    equ_info['listDate'] = [x + timedelta(90) for x in equ_info['listDate']]
    for sec in df.columns:
        if sec[0] not in '036':
            continue
        sec_info = equ_info.ix[sec]
        df.loc[:sec_info['listDate'], sec] = np.NaN
        df.loc[sec_info['delistDate']:, sec] = np.NaN

    return df
"""