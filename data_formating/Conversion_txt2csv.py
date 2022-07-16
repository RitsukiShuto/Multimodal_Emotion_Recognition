# Created by RitsukiShuto on 2022/06/21.
# txtファイルをcsvに変換
#
import pandas as pd

data_dir = "../data/eval/"
file_name = "intensity"  # TODO: 変更せよ
extention = ".txt"
save_dir = "../data/eval/"

read_text_file = pd.read_csv (data_dir + file_name + extention)     # txtファイルを読み込む

read_text_file.columns = ['fid', 'no', 'emotion','EF01', 'EF02', 'EF03', 'EF04', 'EF06',
                          'EM01', 'EM02', 'EM03', 'EM04', 'EM05', 'EM06', 'EM07', 'EM08',
                          'EM09', 'EM11', 'EM12', 'EM14', 'EM15']   # ヘッダを追加

read_text_file.to_csv (save_dir + file_name + ".csv", index=False)               # 保存