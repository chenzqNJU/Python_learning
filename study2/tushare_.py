import tushare as ts
del ts

import tushare as ts
help(ts)
df = ts.get_tick_data('600848',date='2014-01-09')
# 历史分笔
ts.get_tick_data('600848', date='2014-01-09')

# 当日历史分
ts.get_today_ticks('601333')

# 实时分笔
ts.get_realtime_quotes('000581')

# 历史分笔 和 当日历史分 返回结果
time：时间
price：成交价格
pchange：涨跌幅
change：价格变动
volume：成交手
amount：成交金额(元)
type：买卖类型【买盘、卖盘、中性盘】