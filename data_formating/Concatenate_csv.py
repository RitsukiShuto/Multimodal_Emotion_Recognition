# Created by RitsukiShuto on 2022/06/28.
# csvファイルを連結する
#
import pandas as pd
import glob

import numpy as np

import re

csv_list = glob.glob("../data/trans/*.csv")
join_csv1 = pd.read_csv("../data/eval/category.csv", encoding="UTF-8")
join_csv2 = pd.read_csv("../data/eval/intensity.csv", encoding="UTF-8")

join_csv2 = join_csv2.drop(columns=['EF01', 'EF02', 'EF03', 'EF04', 'EF06', 'EM01',
                                    'EM02', 'EM03', 'EM04', 'EM05', 'EM06', 'EM07',
                                    'EM08', 'EM09', 'EM11', 'EM12', 'EM14', 'EM15'], axis=1)

data_list = []

# csvをユニオン
for file in csv_list:
    print(file)
    data_list.append(pd.read_csv(file))

df = pd.concat(data_list, axis=0, sort=False)

df = pd.merge(df, join_csv1, on=['fid', 'no'], how='outer')    # csvをジョイン
df = pd.merge(df, join_csv2, on=['fid', 'no'], how='outer')
#df.sort_values(['fid'])      # csvを並べ替え

df.to_csv("../data/supervised_list.csv", index=False)