# Created by RitsukiShuto on 2022/07/12.
#
#
import pandas as pd

import glob
import os

save_dir = "../data/trans/"
csv_list = glob.glob("../OGVC/OGVC_Vol1/Natural/trans/*.txt")

for csv_file in csv_list:
    print(csv_file)
    df = pd.read_csv(csv_file, header=None)

    file_name, ext = os.path.splitext( os.path.basename(csv_file))
    #print("save to", file_name)        # DEBUG

    df.columns = ['no', 'start', 'end', 'person', 'text']   # ヘッダを追加
    df['fid'] = file_name
    df = df.reindex(columns=['fid', 'no', 'start', 'end', 'person', 'text'])

    df.to_csv (save_dir + file_name + ".csv", index=False)