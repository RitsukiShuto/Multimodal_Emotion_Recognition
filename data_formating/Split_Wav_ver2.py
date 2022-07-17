# Created by RitsukiShuto on 2022/06/28.
# 発話ごとにwavファイルを分割する。その際、{笑}のみの音声は除外する。
#
from cmath import nan
from pydub import AudioSegment

import csv
import pandas as pd

wav_dir = "../OGVC/OGVC_Vol1/Natural/wav/"
csv_dir = "../data/supervised_list.csv"
labeled_dir = "../data/wav/labeled/"
un_labeled_dir = "../data/wav/un_labeled/"

def split_wav(row, path):
    wav_data = AudioSegment.from_file(wav_dir + str(row[0]) + ".wav")

    start = float(row[2]) * 1000
    end = float(row[3]) * 1000

    split = wav_data[start:end]     # wavファイルを分割
    split.export(path + str(row[0]) + "_" + str(row[1]) + ".wav", format="wav")     # 保存


def main():
    csv_file = pd.read_csv(csv_dir, encoding='UTF-8', header=0)

    for row in csv_file.values:
        print(row[6])

        if pd.isnull(row[9]):
            # split_wav(row, un_labeled_dir)
            print("UN LABELED")
        else:
            split_wav(row, labeled_dir)
            print("LABELED")
            
            

if __name__ == "__main__":
    main()