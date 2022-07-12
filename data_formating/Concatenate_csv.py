# Created by RitsukiShuto on 2022/06/28.
# csvファイルを連結する
#
import pandas as pd
import glob

csv_list = glob.glob("../data/trans/*.csv")
join_csv = pd.read_csv("../data/eval/category.txt", encoding="UTF-8")

for i in csv_list:
    print(i)    # DEBUG

data_list = []

for file in csv_list:
    data_list.append(pd.read_csv(file))

df = pd.concat(data_list, axis=0, sort=False)
df.drop(columns = ['Unnamed: 0'], axis=1, inplace=True)

#df_marged = pd.merge(df, join_csv, how='right')

df.to_csv("../data/supervised_list.csv", index=False)