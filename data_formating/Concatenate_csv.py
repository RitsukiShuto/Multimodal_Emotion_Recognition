# Created by RitsukiShuto on 2022/06/28.
# csvファイルを連結する
#
import pandas as pd
import glob

csv_list = glob.glob("../train_data/trans/*.csv")

for i in csv_list:
    print(i)    # DEBUG

data_list = []

for file in csv_list:
    data_list.append(pd.read_csv(file))

df = pd.concat(data_list, axis=0, sort=False)
df.drop(columns = ['Unnamed: 0', 'No', 'start', 'end', 'person'], axis=1, inplace=True)

df.to_csv("../train_data/trans/text-only.csv", index=False)