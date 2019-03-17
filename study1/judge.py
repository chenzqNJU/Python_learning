#! -*-encoding:utf-8 -*-
# coding = utf-8
def whichEncode(text):
  text0 = text[0]
  try:
    text0.decode('utf8')   # decode 转成unicode
  except Exception, e:
    if "unexpected end of data" in str(e):
      return "utf8"
    elif "invalid start byte" in str(e):
      return "gbk_gb2312"
    elif "ascii" in str(e):
      return "Unicode"
  return "utf8"
if __name__ == "__main__":
  print(whichEncode(u"啊".encode("gbk")))
  print(whichEncode(u"啊".encode("utf8")))
  print(whichEncode(u"啊"))



  def getCoding(strInput):
    '''
    获取编码格式
    '''
    if isinstance(strInput, unicode):
        return "unicode"
    try:
        strInput.decode("utf8")
        return 'utf8'
    except:
        pass
    try:
        strInput.decode("gbk")
        return 'gbk'
    except:
        pass
getCoding(data)

def tran2UTF8(strInput):
    '''
    转化为utf8格式
    '''
    strCodingFmt = getCoding(strInput)
    if strCodingFmt == "utf8":
        return strInput
    elif strCodingFmt == "unicode":
        return strInput.encode("utf8")
    elif strCodingFmt == "gbk":
        return strInput.decode("gbk").encode("utf8")

def tran2GBK(strInput):
    '''
    转化为gbk格式
    '''
    strCodingFmt = getCoding(strInput)
    if strCodingFmt == "gbk":
        return strInput
    elif strCodingFmt == "unicode":
        return strInput.encode("gbk")
    elif strCodingFmt == "utf8":
        return strInput.decode("utf8").encode("gbk")

fr = open('e:\\python\\tip.txt','r')   #读写模式：r-只读；r+读写；w-新建（会覆盖原有文件）;更多信息见参考文章2
all_utf8 = fr.read()
print (all_utf8)
all_uni = all_utf8.decode("utf-8")
all_uni = unicode(all_utf8, 'utf-8')

isinstance('哈哈你好',str)
a=u'地方'
print (u'地方')
print (a.decode('UTF-8'))
print (u"拟合")