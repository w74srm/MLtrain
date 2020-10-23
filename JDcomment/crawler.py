import requests
import json
import time
import pandas as pd
import numpy as np
import re
import random

findJSON = re.compile(r'fetchJSON_comment98\((.*)\);')

proxies = {'http': ['http://110.243.16.155:9999',
                    'http://113.194.49.146:9999',
                    'http://175.42.129.26:9999',
                    'http://163.204.242.20:9999',
                    'http://182.34.36.65:9999',
                    'http://112.64.233.130:9991',
                    'http://139.159.7.150:52908']}


JD_Id = ['100006487373', '100007218425', '100012445728', '100007852387', '100007270267', '100006546527', '100013312610', '100011773072', '100007539330', '12579070740']


def main():
    datalist = []
    cnt = 0
    baseurl = "https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId={}&score={}&sortType=5&page={}&pageSize=10&isShadowSku=0&rid=0&fold=1"
    savepath = "jdcomment.csv"
    for k in range(10):
        for i in range(1, 4):
            for j in range(100):
                url = baseurl.format(JD_Id[k], i, j)
                print(url)
                rep = askurl(url)
                print(rep)
                try:
                    jsonstr = re.findall(findJSON, rep)[0]
                    src_data = json.loads(jsonstr)
                    src_datalist = src_data['comments']
                    for item in src_datalist:
                        data = []
                        data.append(item["content"])
                        data.append(item["score"])
                        datalist.append(data)
                        cnt += 1
                        print(cnt)
                except IndexError as e:
                    pass
                time.sleep(2 + random.random())

    datalist = np.array(datalist)
    datalist = pd.DataFrame(datalist)
    datalist.to_csv(savepath)


def askurl(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36"
    }
    try:
        response = requests.request(url=url, headers=headers, method='get', proxies=proxies)
    except requests.HTTPError as e:
        pass
    return response.text


if __name__ == '__main__':
    main()
