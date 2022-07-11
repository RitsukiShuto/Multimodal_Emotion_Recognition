# Created by RitsukiShuto on 2022/06/21.
# txtファイルをcsvに変換
#
import pandas as pd

data_dir = "../OGVC/OGVC_Vol1/Natural/trans/"
file_name = "06_FWA"  # TODO: 変更せよ
extention = ".txt"
save_dir = "../train_data/trans/"

read_text_file = pd.read_csv (data_dir + file_name + extention)     # txtファイルを読み込む

read_text_file.columns = ['No', 'start', 'end', 'person', 'text']   # ヘッダを追加
read_text_file.to_csv (save_dir + file_name + ".csv")               # 保存