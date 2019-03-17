import time
import datetime

#把datetime转成字符串
def datetime_toString(dt):
    return dt.strftime("%Y-%m-%d-%H")

#把字符串转成datetime
def string_toDatetime(string):
    return datetime.strptime(string, "%Y-%m-%d-%H")

#把字符串转成时间戳形式
def string_toTimestamp(strTime):
    return time.mktime(string_toDatetime(strTime).timetuple())

#把时间戳转成字符串形式
def timestamp_toString(stamp):
    return time.strftime("%Y-%m-%d-%H", time.localtime(stamp))

#把datetime类型转时间戳形式
def datetime_toTimestamp(dateTim):
    return time.mktime(dateTim.timetuple())

datetime_toString(z)

z.timetuple()
time.mktime(z.timetuple())
datetime.datetime.strftime("%Y-%m-%d-%H",time.localtime(1358006400))
z.strftime("%Y-%m-%d-%H")

now = datetime.datetime.now()

# 前一小时
d1 = now - datetime.timedelta(hours=1)
print d1.strftime("%Y-%m-%d %H:%S:%M")

# 前一天
d2 = now - datetime.timedelta(days=1)
print d2.strftime("%Y-%m-%d %H:%S:%M")

# 上周日
d3 = now - datetime.timedelta(days=now.isoweekday())
print d3.strftime("%Y-%m-%d %H:%S:%M"), " ", d3.isoweekday()

# 上周一
d31 = d3 - datetime.timedelta(days=6)
print d31.strftime("%Y-%m-%d %H:%S:%M"), " ", d31.isoweekday()

# 上个月最后一天
d4 = now - datetime.timedelta(days=now.day)
print d3.strftime("%Y-%m-%d %H:%S:%M")

# 上个月第一天
print datetime.datetime(d4.year, d4.month, 1) 