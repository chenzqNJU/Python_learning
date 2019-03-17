import tushare as ts
import numpy as np
import pandas as pd
import os
import gc
import myfunc
os.path
os.chdir("e:\\stock")  # 修改当前工作目录

mainData_book = pd.read_excel('e:/wind/ASHAREMANAGEMENTHOLDREWARD.xlsx')

a = pd.read_csv('e:/wind/ASHAREMANAGEMENTHOLDREWARD.csv',encoding='gb2312',header=None,ignore=True)


b=pd.read_sas('e:\\wind\\hold.sas7bdat')
# b.drop('VAR3',axis=1,inplace=True)

b.VAR2 = b.VAR2.str.decode('utf8')
#str(b.VAR2, encoding = "utf8")
b.VAR4=b.VAR4.astype('int').astype(str)

# 选出 6.30  12.31 的
b['date'] = pd.to_datetime(b.VAR4,format='%Y%m%d')

b1=b[b.VAR4.str.contains('0630') | b.VAR4.str.contains('1231')]
b1 = b1.iloc[:,:3]
b1.to_csv('e:/wind/manager.csv',index=False)
myfunc.search('to_csv')

help(pd.DataFrame.to_csv)
len(b)
len(b1)


t=['000650.XSHE', '002014.XSHE', '002088.XSHE', '002107.XSHE', '002117.XSHE', '002136.XSHE', '002148.XSHE', '002270.XSHE', '002365.XSHE', '002561.XSHE', '002553.XSHE', '002706.XSHE', '002718.XSHE', '002732.XSHE', '002749.XSHE', '002763.XSHE', '002793.XSHE', '002790.XSHE', '002836.XSHE', '002853.XSHE', '002849.XSHE', '002900.XSHE', '002880.XSHE', '002887.XSHE', '002896.XSHE', '002913.XSHE', '002868.XSHE', '002882.XSHE', '300075.XSHE', '300081.XSHE', '300231.XSHE', '300258.XSHE', '300286.XSHE', '300295.XSHE', '300341.XSHE', '300371.XSHE', '300396.XSHE', '300445.XSHE', '300410.XSHE', '300428.XSHE', '300446.XSHE', '300427.XSHE', '300394.XSHE', '300547.XSHE', '300519.XSHE', '300548.XSHE', '300561.XSHE', '300600.XSHE', '300637.XSHE', '300635.XSHE', '300648.XSHE', '300656.XSHE', '300687.XSHE', '300590.XSHE', '300599.XSHE', '300605.XSHE', '300695.XSHE', '300701.XSHE', '300696.XSHE', '300709.XSHE', '300727.XSHE', '300735.XSHE', '300723.XSHE', '300707.XSHE', '300739.XSHE', '300711.XSHE', '300738.XSHE', '300732.XSHE', '600305.XSHG', '600345.XSHG', '600419.XSHG', '600668.XSHG', '600732.XSHG', '600844.XSHG', '600892.XSHG', '600995.XSHG', '603023.XSHG', '603037.XSHG', '603158.XSHG', '603266.XSHG', '603086.XSHG', '603079.XSHG', '603110.XSHG', '603226.XSHG', '603106.XSHG', '603136.XSHG', '603283.XSHG', '603058.XSHG', '603599.XSHG', '603339.XSHG', '603306.XSHG', '603585.XSHG', '603535.XSHG', '603326.XSHG', '603500.XSHG', '603496.XSHG', '603608.XSHG', '603639.XSHG', '603808.XSHG', '603669.XSHG', '603809.XSHG', '603829.XSHG', '603722.XSHG', '603607.XSHG', '603617.XSHG', '603856.XSHG', '603889.XSHG', '603987.XSHG', '603968.XSHG', '603928.XSHG', '603978.XSHG', '603938.XSHG', '603960.XSHG']
len(t)

############################## 吉姆．史莱特 (Jim Slater) 祖鲁原则投资法 ##############################
# 选股标准：
# 1.	总市值 < 市场平均总市值*1.0。
# 2.	过去五年税后净利皆为正值。
# 3.	过去三年税后净利成长率皆 >= 15%。
# 4.	预估税后净利成长率 >= 15%。
# 5.	近五年平均现金流量 > 近五年平均税后净利。
# 6.	近四季现金流量 > 最近四季税后净利。
# 7.	近四季营业利益率 >= 10%。
# 8.	近四季可运用资本报酬率 >= 10%。
# 9.	最近一季负债/净值比 < 50%。
# 10.	最新董监事持股比例 >= 20%　或　最新一期董监事持股比例增加。
# 　买进标准：
# 1.	预估本益比 <= 20。
# 2.	预估本益比/成长率比值(PEG) <= 1.2。
# 3.	最近一年股价相对强弱势(RS) > 1。
# 4.	最近一个月股价相对强弱势(RS) > 1。

import pandas as pd
import numpy as np
from CAL.PyCAL import *
from pandas import DataFrame, Series
from datetime import datetime, timedelta

cal = Calendar('China.SSE')
start = '2010-05-01'
end = '2018-05-13'
universe = DynamicUniverse('A')
benchmark = 'SHCI'
freq = 'd'
refresh_rate = Monthly(1)

accounts = {
    'fantasy_account': AccountConfig(account_type='security', capital_base=10000000)
}
max_history_window = 200


def initialize(context):
    pass


def handle_data(context):
    date = context.current_date
    if date.month in [5, 9, 11]:
        global temp, temp1
        account = context.get_account('fantasy_account')
        previous_date = context.previous_date.strftime('%Y%m%d')
        yester_30day = cal.advanceDate(context.current_date, '-30B', BizDayConvention.Following).strftime('%Y%m%d')
        yester_1year = cal.advanceDate(context.current_date, '-1Y', BizDayConvention.Following).strftime('%Y%m%d')
        yester_5year = cal.advanceDate(context.current_date, '-5Y', BizDayConvention.Following).strftime('%Y%m%d')
        yester_1month = cal.advanceDate(context.current_date, '-1M', BizDayConvention.Following).strftime('%Y%m%d')

        stk = context.get_universe(exclude_halt=True)

        ## 1.	总市值<均值  预估本益比 <= 20 预估本益比/预估盈余成长率(PEG) < 1.2 预估税后净利成长率 >= 15%  近四季营业利益率 >= 10%  近四季现金流量 > 最近四季税后净利 	近四季可运用资本报酬率 >= 10%  最近一季负债/净值比 < 50%。
        data = DataAPI.MktStockFactorsOneDayGet(tradeDate=previous_date, secID=stk,
                                                field=u"secID,LCAP,FY12P,FEARNG,OperatingProfitRatio,ETOP,PCF,ROA,LongTermDebtToAsset,EquityToAsset,DebtsAssetRatio",
                                                pandas="1")
        # 其中，FEARNG 预估盈余成长率  FY12P 预估盈余/市值 = 1/预估本益比 OperatingProfitRatio 营业利润率ttm
        # DebtsAssetRatio = 总负债 / 总资产

        data['rate'] = 1 / (data.FY12P * data.FEARNG)  # 预估本益比/预估盈余成长率(PEG)
        data['cash'] = data.ETOP * data.PCF  # (净利润ttm/总市值) * (总市值/现金流量净额ttm)
        data['NIncomeTTM'] = data.ETOP * np.exp(data.LCAP)  # 净利润ttm = ETOP * 总市值
        data['Asset'] = data.NIncomeTTM / data.ROA  # 总资产 = 净利润ttm / (净利润ttm/总资产)
        data['Longdebt'] = data.LongTermDebtToAsset * data.Asset  # 长期负债
        data['Equity'] = data.EquityToAsset * data.Asset  # 股东权益
        data['capital_employedTTM'] = data.NIncomeTTM / (data.Longdebt + data.Equity)
        # 可运用资本报酬率ttm = 净利润ttm /（长期负债+股东权益）
        data['DebtsNetAssetRatio'] = data.DebtsAssetRatio / (1 - data.DebtsAssetRatio)  # 负债净值比 = 总负债/（总资产-总负债）

        # list1 = data[(data.FEARNG >= 0.15) & (data.OperatingProfitRatio >= 0.10) & (data.cash < 1) & (data.capital_employedTTM > 0.1) & (data.DebtsNetAssetRatio < 0.5) & (data.LCAP <= data.LCAP.median()) & (data.FY12P >= 1/20) &(( (data.rate <= 200) & (data.rate > 0))|data.rate.isnull())]
        list1 = data[(((data.rate <= 120) & (data.rate > 0)) | data.rate.isnull()) & (data.DebtsNetAssetRatio < 0.5) & (
                data.LCAP <= data.LCAP.mean()) & ((data.FY12P >= 1 / 20) | (data.FY12P.isnull())) & (
                             (data.FEARNG >= 0.15) | (data.FEARNG.isnull())) & (data.OperatingProfitRatio >= 0.10) & (
                             data.cash < 1) & (data.capital_employedTTM > 0.1)]
        list1 = list1.secID.tolist()
        # print data

        ## 2.	最近 1年 and 1月 股价相对强弱势(RS) > 1
        mv1 = DataAPI.MktEqudGet(secID=stk, tradeDate=yester_1year, field=u"secID,tradeDate,marketValue", pandas="1")
        mv2 = DataAPI.MktEqudGet(secID=stk, tradeDate=yester_1month, field=u"secID,tradeDate,marketValue", pandas="1")
        mv3 = DataAPI.MktEqudGet(secID=stk, tradeDate=previous_date, field=u"secID,tradeDate,marketValue", pandas="1")

        mv = pd.concat([mv1, mv2, mv3]).pivot(index='secID', columns='tradeDate', values='marketValue')
        mv = mv.apply(lambda x: x / x.sum(), axis=0)
        mv1 = mv[(mv.iloc[:, 0] < mv.iloc[:, 2]) & (mv.iloc[:, 1] < mv.iloc[:, 2])]

        list2 = mv1.index.tolist()

        ## 3. 过去五年税后净利皆为正值  近五年平均现金流量 > 近五年平均税后净利 过去三年税后净利成长率皆 >= 15%。
        I = DataAPI.FdmtISGet(secID=stk, reportType=u"A", endDate=previous_date, beginDate=yester_5year,
                              field=u"secID,endDate,reportType,NIncome", pandas="1")
        I.drop_duplicates(['secID', 'endDate'], inplace=True)
        C = DataAPI.FdmtCFGet(secID=stk, reportType=u"A", endDate=previous_date, beginDate=yester_5year,
                              field=u"secID,endDate,reportType,NCFOperateA", pandas="1")
        C.drop_duplicates(['secID', 'endDate'], inplace=True)
        c = C.groupby('secID')['NCFOperateA'].mean()  # 计算近5年平均现金流量
        a = I.pivot(index='endDate', columns='secID', values='NIncome')
        b = a.mean()  # 计算近5年平均净利润
        tem = pd.concat([c, b.rename('NIncome')], axis=1)

        t1 = tem[tem.NCFOperateA > tem.NIncome].index.tolist()  # 平均现金流量 > 平均净利润

        a = a[a > 0].dropna(how='any', axis=1)  # 删除近5年净利润有亏损的股票
        a = a.pct_change()[-3:]  # 计算近3年的净利润增长率
        a = a[a > 0.05].dropna(how='any', axis=1)  # 删除近3年增长率有不足15%的股票
        t2 = a.columns.tolist()

        list3 = list(set(t1) & set(t2))

        print
        list1
        print
        list2
        print
        list3

        buylist = list(set(list1) & set(list2) & set(list3))
        print
        buylist

        position = account.get_positions()
        buy_list = buylist
        # 判断持仓是否为空
        if len(position) > 0:
            # 获取停牌secid
            notopen = DataAPI.MktEqudGet(tradeDate=context.now, secID=position.keys(), isOpen="0", field=u"secID",
                                         pandas="1")
            sum_ = 0
            # 计算停牌secID的权益
            for sec in notopen.secID:
                tmp = account.get_position(sec).value
                sum_ += tmp
            buyweight = 1.0 - sum_ / account.portfolio_value
        else:
            buyweight = 1.0
        for stk in position:
            # 先卖
            if stk not in buy_list:
                account.order_to(stk, 0)
        if len(buy_list) > 0:
            weight = buyweight / len(buy_list)
        else:
            weight = 0
        for stk in buy_list:
            if stk in account.get_positions():  # 先对手里有的票进行调仓
                account.order_pct_to(stk, weight)
        for stk in buy_list:
            if stk not in account.get_positions():  # 再对这次要买入的进行调仓
                account.order_pct_to(stk, weight)

#          # 交易部分
#         positions = account.get_positions()
#         sell_list = [stk for stk in positions if stk not in buylist]
#         for stk in sell_list:
#             account.order_to(stk, 0)

#         c = account.portfolio_value
#         change = {}

#         for stock in buylist:
#             p = context.current_price(stock)

#             if not np.isnan(p) and p > 0:
#                 if stock in positions:
#                     change[stock] = int(c / len(buylist) / p) - positions[stock].amount
#                 else:
#                     change[stock] = int(c / len(buylist) / p) - 0

#         for stock in sorted(change, key=change.get):
#             account.order(stock, change[stock])
