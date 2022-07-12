# Created by RitsukiShuto on 2022/06/28.
# csvファイルを連結する
#
import pandas as pd
import glob

import numpy as np

import re

csv_list = glob.glob("../data/trans/*.csv")
join_csv = pd.read_csv("../data/eval/category.csv", encoding="UTF-8")

for i in csv_list:
    print(i)    # DEBUG

data_list = []

# csvをユニオン
for file in csv_list:
    data_list.append(pd.read_csv(file))

df = pd.concat(data_list, axis=0, sort=False)

df = pd.merge(df, join_csv, on=['fid', 'no'], how='outer')    # csvをジョイン
#df.sort_values(['fid'])      # csvを並べ替え

df.to_csv("../data/supervised_list.csv", index=False)