import urllib.request
import json
import csv
import datetime
import pytz
import time
import concurrent.futures
import schedule
import io


def match_twins(month: int):
    prefix = 'http://hq.sinajs.cn/list=OP_'
    suffix = '_510050'
    url1 = f'{prefix}UP{suffix}{str(month)}'
    url2 = f'{prefix}DOWN{suffix}{str(month)}'
    return (get_paried_urls([url1, url2]))


def get_paried_urls(twin_list: list) -> list:
    urls = []
    paired_url = []
    for url in twin_list:
        content = urllib.request.urlopen(url, None).read().decode('GBK')
        paired_url.append(get_all_name(content))
    return (re_pair(paired_url))


def get_all_name(content) -> list:
    quo_pos = content.find('"')
    seg = content[quo_pos + 1:-3]
    stock_list = seg.split(',')
    return stock_list[:-1]


def re_pair(li) -> list:
    finished_pair = []
    for i in range(len(li[0])):
        middle_pair = []
        middle_pair.append(li[0][i])
        middle_pair.append(li[1][i])
        finished_pair.append(middle_pair)

    return finished_pair


######### PAIR to DATA
# DATA = match_twins(num)
def data_parser(query):
    prefix = 'http://hq.sinajs.cn/list='

    url = prefix + query
    data = urllib.request.urlopen(url, None).read().decode('GBK')

    eq_pos = data.find('=')
    params_seg = data[eq_pos + 2:-3]
    params = params_seg.split(',')

    return ([query] + params)


# url->
def get_expire_url(month) -> str:
    prefixDate = 'http://stock.finance.sina.com.cn/futures/api/openapi.php/StockOptionService.getRemainderDay?date='
    url = f'{prefixDate}{str(month)}'
    return url


def get_expire_date(url_link) -> str:
    with urllib.request.urlopen(url_link) as url:
        data = json.loads(url.read().decode())

        return (data['result']['data']['expireDay'])


def get_date_string(i) -> str:
    return ''.join((datetime.date.today() +
                    datetime.timedelta(i * 365 / 12)).isoformat().split('-'))


def pair_to_list(pairs) -> list:
    res = []
    for pair in pairs:
        res.extend(pair)
    return res


def write_data(i):
    date_string = get_date_string(i)

    if len(match_twins(date_string[2:6])) == 0:
        # print(f'no data found in {date_string[4:6]} 月')
        pass
    else:
        with concurrent.futures.ThreadPoolExecutor(16) as tp:
            future_row = {
                tp.submit(data_parser, item): item
                for item in pair_to_list(match_twins(date_string[2:6]))
            }

        for future in concurrent.futures.as_completed(future_row):
            item = future_row[future]

            try:
                res = future.result()
            except Exception as e:
                print('Exception when processing item "%s": %s' % (item, e))
            else:
                # print(res)
                writer.writerow([date_string] + [
                    datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
                ] + res)


def job():
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(12) as pool:
        future_writers = {pool.submit(write_data, i): i for i in range(12)}

    print("--- %s seconds ---" % (time.time() - start_time))


#### Writing to CSV
with io.open(
        'sing_stock_data.csv', 'a', newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile, delimiter=',')

    schedule.every(1).seconds.do(job)

    while True:
        schedule.run_pending()
        time.sleep(1)
