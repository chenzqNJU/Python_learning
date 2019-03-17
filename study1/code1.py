# -*- coding: utf-8 -*-
"""
作者：邢不行

本系列帖子“量化小讲堂”，通过实际案例教初学者使用python、pandas进行金融数据处理，希望能对大家有帮助。

必读文章《10年400倍策略分享-附视频逐行讲解代码》：http://bbs.pinggu.org/thread-5558776-1-1.html

所有系列文章汇总请见：http://bbs.pinggu.org/thread-3950124-1-1.html

想要快速、系统的学习量化知识，可以参与我与论坛合作开设的《python量化投资入门》视频课程：http://www.peixun.net/view/1028.html，我会亲自授课，随问随答。
参与课程还可以免费加入我的小密圈，我每天会在圈中分享量化的所见所思，圈子介绍：http://t.xiaomiquan.com/BEiqzVB

微信：xbx_laoshi，量化交流Q群(快满)：438143420，有问题欢迎交流。

文中用到的A股数据可在www.yucezhe.com下载，这里可以下载到所有股票、从上市日起的交易数据、财务数据、分钟数据、分笔数据、逐笔数据等。
"""
import pandas as pd


# =====导入月线数据
# 注意：filepath_or_buffer参数中请填写数据在你电脑中的路径
stock_data = pd.read_csv(filepath_or_buffer='stock_data.csv', parse_dates=[u'交易日期'], encoding='gbk')

# =====计算每个股票下个月的涨幅
stock_data[u'下个月涨跌幅'] = stock_data.groupby(u'股票代码')[u'涨跌幅'].shift(-1)

# =====删除一些不满足条件的股票数据
# 删除在某些月份市净率小于0的股票
stock_data = stock_data[stock_data[u'市净率'] > 0]
# 删除在当月最后一个交易日停牌的股票
stock_data = stock_data[stock_data[u'是否交易'] == 1]
# 删除在当月最后一个交易日涨停的股票
stock_data = stock_data[stock_data[u'是否涨停'] == 0]
# 删除在本月交易日小于10天的股票
stock_data = stock_data[stock_data[u'交易天数'] > 10]
# 删除2000年之前的股票数据
stock_data = stock_data[stock_data[u'交易日期'] > pd.to_datetime('20000101')]
# 删除"下个月涨跌幅"字段为空的行
stock_data.dropna(subset=[u'下个月涨跌幅'], inplace=True)

# =====选股
stock_data[u'factor'] = stock_data[u'总市值'] * stock_data[u'市净率']
stock_data = stock_data.sort([u'交易日期', u'factor'])  # 排序
stock_data = stock_data.groupby([u'交易日期']).head(10)  # 选取前十名的股票

# =====计算每月选股收益
output = pd.DataFrame()
stock_data[u'股票代码'] += u' '
stock_data_groupby = stock_data.groupby(u'交易日期')
output[u'买入股票'] = stock_data_groupby[u'股票代码'].sum()
output[u'股票数量'] = stock_data_groupby.size()
output[u'买入股票下月平均涨幅(%)'] = stock_data_groupby[u'下个月涨跌幅'].mean()
output[u'下个月末的资金(初始100)'] = (output[u'买入股票下月平均涨幅(%)']+1.0).cumprod() * 100.0
output.reset_index(inplace=True)

# ======输出
output.to_csv('output.csv', index=False, encoding='gbk')
