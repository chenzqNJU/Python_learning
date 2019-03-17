from pandas import DataFrame
from other.dataapi_win36  import Client
import json
client = Client()
client.init('62e0be4c252d72679c4e90bf86398c0895af8ff46d23d1e53923d740252b466b')
url='/api/equity/getEqu.json?field=ticker,secShortName,listDate,delistDate&listStatusCD=L,S,DE,UN&secID=&ticker=&equTypeCD=A'
code, result = client.getData(url)
j = json.loads(result.decode())
d = DataFrame(j['data'])
d = d.set_index('ticker')
d = d[['secShortName','listDate','delistDate']]
d.to_csv('data/ticker_and _day_of_(de)list_date.csv')

protocol：https method：“GET” url：“https://api.datayes.com:443/data/market/getHistoryTicksOneDay.json?field=lastPrice&s header：“Authorization: Bearer <access_token>”


def getIdxHq():
    client = Client()
    client.init('开发者凭证')
    url2 = "/api/market/getMktIdxd.json?field=&beginDate=20060101&endDate=20150930&indexID=&ticker=000300&tradeDate="
    code, result = client.getData(url2)

    if code == 200:
        print
        result
        obj = json.loads(result)
        a = obj['data']
        str = "insert into index_hq values"
        ###此时返回的数据a为list，其中的元素为字典
        for i in a:
            str = str + "(\"%s\",\"%s\", \"%s\",\"%s\",\"%s\",\"%s\",%f,%f,%f,%f,%f,%f,%f,%f,%f);\n" % (i["indexID"],

                                                                                                        i["tradeDate"],
                                                                                                        i["ticker"], "",
                                                                                                        "",
                                                                                                        i["exchangeCD"],
                                                                                                        i[
                                                                                                            "preCloseIndex"],
                                                                                                        i["openIndex"],
                                                                                                        i[
                                                                                                            "lowestIndex"],
                                                                                                        i[
                                                                                                            "highestIndex"],
                                                                                                        i["closeIndex"],
                                                                                                        i[
                                                                                                            "turnoverVol"],
                                                                                                        i[
                                                                                                            "turnoverValue"],
                                                                                                        i["CHG"],
                                                                                                        i["CHGPct"])
            str = str + "insert into index_hq values"

        fileobj = codecs.open("index_hq000300.sql", 'w', "utf-8")
        fileobj.write(str)
        fileobj.close()

    else:
        print
        code
        print
        result