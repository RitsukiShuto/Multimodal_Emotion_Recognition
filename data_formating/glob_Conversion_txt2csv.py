# Created by RitsukiShuto on 2022/07/12.
#
#
import pandas as pd

import glob
import os

save_dir = "../data/trans/OGVC_vol2/"
csv_list = glob.glob("../OGVC/OGVC_Vol2/Acted/text/dlg*.txt")

for csv_file in csv_list:
    print(csv_file)
    df = pd.read_csv(csv_file, header=None)

    df.columns = ['no', 'person', 'emotion', 'text']   # ヘッダを追加
    df = df.reindex(columns=['no', 'person', 'emotion', 'text'])

    file_name, ext = os.path.splitext( os.path.basename(csv_file))
    df.to_csv (save_dir + file_name + ".csv", index=False)