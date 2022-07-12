# Created by RitsukiShuto on 2022/06/21.
# txtファイルをcsvに変換
#
import pandas as pd

data_dir = "../data/eval/"
file_name = "category"  # TODO: 変更せよ
extention = ".txt"
save_dir = "../data/eval/"

read_text_file = pd.read_csv (data_dir + file_name + extention)     # txtファイルを読み込む

read_text_file.columns = ['fid', 'No', 'ans1', 'ans2', 'ans3']   # ヘッダを追加
read_text_file.to_csv (save_dir + file_name + ".csv", index=False)               # 保存