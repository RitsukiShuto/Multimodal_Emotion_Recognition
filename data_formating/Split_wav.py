# Created by RitsukiShuto on 2022/06/21.
# csvファイルを参照してwavファイルを発話ごとに分割する。<試作版>
#
from pydub import AudioSegment
from pydub.silence import split_on_silence

import csv
import os

data_dir = "../OGVC/OGVC_Vol1/Natural/wav/"
file_name = "06_FWA"      # TODO: 変更せよ
extention = ".wav"
save_dir = "../train_data/sound/06_FWA/"      # TODO: 変更せよ

# wavファイルを取得
wav_data = AudioSegment.from_file(data_dir + file_name + extention)

# csvファイルを取得
csv_data = open("../train_data/trans/06_FWA.csv", "r", encoding="utf-8", errors="", newline="")       # TODO: 変更せよ
f = csv.reader(csv_data, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
header = next(f)    # ヘッダをスキップ

# ディレクトリを作成
os.makedirs(save_dir)

# DEBUG
print("Editting", wav_data, ".")

# wavファイルを分割して保存
for row in f:
    start = float(row[2]) * 1000      # 秒をミリ秒に変換
    end = float(row[3]) *1000         # 同上

    split = wav_data[start:end]     # wavファイルを分割
    print(split)        # DEBUG

    split.export(save_dir + file_name + "_" + row[0] + extention, format="wav")     # 保存