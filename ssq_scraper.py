import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime

# 目标URL
url = f'http://datachart.500.com/ssq/history/newinc/history.php?start=07001'

# 发送HTTP请求
response = requests.get(url)
response.encoding = 'utf-8'  # 确保编码正确

# 解析HTML内容
soup = BeautifulSoup(response.text， 'html.parser')

# 定位包含开奖数据的表格体
tbody = soup.find('tbody', id="tdata")

# 存储开奖数据的列表
lottery_data = []

# 遍历每一行数据
for tr in tbody.find_all('tr'):
    tds = tr.find_all('td')
    if tds:
        # 提取数据并添加到列表
        lottery_data.append([td.text for td in tds])

# 写入CSV文件
csv_filename = 'ssq_lottery_data.csv'
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # 写入标题行
    # writer.writerow(['期号', '号码1', '号码2', '号码3', '号码4', '号码5', '号码6', '号码7'])
    # 写入数据行
    writer.writerows(lottery_data)

# 写入日志文件
log_filename = 'log.txt'
with open(log_filename, 'a', encoding='utf-8') as logfile:
    logfile.write(f'{datetime.当前()} - 数据抓取完成，并保存到 {csv_filename} 文件中。\n')

print('数据抓取完成，并保存到ssq_lottery_data.csv文件中。')
