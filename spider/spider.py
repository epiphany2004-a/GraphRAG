import os
import time

import pandas as pd
import requests
import jsonpath
import threading
from concurrent.futures import ThreadPoolExecutor
from lxml import etree
import json

headers = {
    # 此处的cookie为翻页时候提取的cookie
    "Cookie": "wdcid=2c1008ae99960cdf; UM_distinctid=17cdb56878c89c-0200b6372d3363-a7d173c-186a00-17cdb56878d9b7; __asc=f667c92517cdb5688ae3ad4667f; __auc=f667c92517cdb5688ae3ad4667f",
    "Host": "newssearch.chinadaily.com.cn",
    "Referer": "http://newssearch.chinadaily.com.cn",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest"
}


def get_resp(url):
    """根据翻页url提取信息，为json格式"""
    response = requests.get(headers=headers, url=url)
    print(response.text)
    return response.json()


def format_data(data):
    """
    提取url两种方案，因为目前发现根据网页风格url至少存在两种提取方案
    :param data:
    :return:
    """
    # 方案1
    url_list = jsonpath.jsonpath(data, "$..shareUrl")

    # 方案2
    url_list_1 = jsonpath.jsonpath(data, "$..url")

    i = 0
    for url in url_list:
        if url is None:  # 如果第一种方案提取不到
            url_list[i] = url_list_1[i]  # 用第二种方案提取到的url替换第一种
        i += 1

    print(url_list)
    return url_list


def save_data(title, pub_time, content, source, author, url, name):
    data_dic = {}
    data_dic["title"] = title
    data_dic["pub_time"] = pub_time
    data_dic["content"] = content
    data_dic["source"] = source
    data_dic["author"] = author
    data_dic["url"] = url

    # 将字典转换为DataFrame
    df = pd.DataFrame([data_dic])

    # 检查文件是否存在
    file_exists = os.path.exists(f'all_data/orginal_data/{name}.csv')

    # 如果文件不存在,创建新文件并写入表头
    # 如果文件存在,追加数据不写入表头
    if not os.path.exists('all_data'):
        os.makedirs('all_data')
    if not os.path.exists('all_data/orginal_data'):
        os.makedirs('all_data/orginal_data')
    df.to_csv(f'all_data/orginal_data/{name}.csv', mode='a', encoding='utf-8', index=False, header=not file_exists)


def get_url_per_detail(url_list, name):
    """
    获取每页具体详细内容
    :param url_list:
    :return:
    """
    headers = {
        "referer": "http://newssearch.chinadaily.com.cn/",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "Windows",
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "cross-site",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36",
        # 此处的cookie为具体请求详情页时候提取的cookie
        "cookie": "wdcid=2c1008ae99960cdf; UM_distinctid=17cdb56878c89c-0200b6372d3363-a7d173c-186a00-17cdb56878d9b7; __auc=f667c92517cdb5688ae3ad4667f; wdses=247a72d46467ad42; U_COOKIE_ID=3b41143ce7a6457f1215daf72548dd40; _ga=GA1.3.1878765827.1635773226; _gid=GA1.3.221562497.1635773226; pt_s_3bfec6ad=vt=1635774957349&cad=; pt_s_747cedb1=vt=1635775097596&cad=; pt_747cedb1=uid=SNll4C41NeyncGhS5-bxuw&nid=0&vid=kwH8z7J3HdhFjuHq2QXdNA&vn=2&pvn=3&sact=1635775106946&to_flag=0&pl=FsrbI2FDvRRKNCdlB12cAA*pt*1635775072479; pt_3bfec6ad=uid=NMM5VwPuB9-chzPwYDVYNg&nid=0&vid=NARFb1eDW/2kCbV3kFI5BQ&vn=3&pvn=1&sact=1635775367977&to_flag=1&pl=aaYpoHDZ8hBm2HkBDOWEGw*pt*1635774957349; wdlast=1635776504; CNZZDATA3089622=cnzz_eid%3D938946984-1635764865-null%26ntime%3D1635767813"
    }

    for url in url_list:
        try:
            resp = requests.get(headers=headers, url=url)
            print(resp.text)
            e = etree.HTML(resp.text)
            # 多规则匹配标题，目前发现两种风格
            title_list = e.xpath("//h1/text() | //h2/text()")
            # 多规则匹配出版时间，目前发现四种风格
            pub_time_list = e.xpath(
                '//div[@class="info"]/span[1]/text() |  //div[@class="content"]/div[1]/p/text() | //div[@class="articl"]//h5/text() | //div[@id="Title_e"]/h6/text()')
            # 多规则匹配内容，目前发现两种风格
            content_list = e.xpath('//div[@id="Content"]/p/text() | //div[@id="Content"]/p[position()<last()]/text()')
            # 只要有一个字段为空，我们就舍弃这条新闻
            if not pub_time_list or not content_list or not title_list:
                continue
            title = title_list[0]
            pub_time = pub_time_list[0].strip().rsplit(": ", 1)[-1]

            content = ""
            for sentence in content_list:
                content += sentence.strip()

            source = ""
            author = pub_time_list[0].strip().split("|")[0]
            pub_time_info = pub_time_list[0].strip().split("|")
            if len(pub_time_info) == 3:  # 格式为 "By YANG RAN | China Daily | Updated: 2024-12-25 11:01"
                source = pub_time_info[1].strip()

            save_data(title=title, pub_time=pub_time, content=content, source=source, author=author, url=url, name=name)

        except Exception as e:
            continue
def spider(name,page1,page2):
        i=1
        with ThreadPoolExecutor(max_workers=5) as executor:
            for page in range(page1, page2):
                print(f"当前正在下载第{i}页......................")
                url = f"http://newssearch.chinadaily.com.cn/rest/en/search?keywords={name}&sort=dp&page={page}&curType=story&type=&channel=&source="

                try:
                    # 获取分页内容，为json格式
                    resp = get_resp(url=url)
                except Exception as e:
                    continue

                # 是否最后一页判断
                if resp.get("code") == 400:
                    break

                # 获取每页的十条url列表
                url_list = format_data(resp)

                # 使用线程池处理每一页具体内容
                executor.submit(get_url_per_detail, url_list, name=name)

                i += 1
                time.sleep(0.3)

if __name__ == '__main__':
    
    name = "INDIA"
    spider(name,0,10)


