# Created by RitsukiShuto on 2022/06/28.
# 発話ごとにwavファイルを分割する。その際、{笑}のみの音声は除外する。
#
from cmath import nan
from pydub import AudioSegment

import csv
import pandas as pd

# data dir
wav_dir = "../OGVC/OGVC_Vol1/Natural/wav/"
csv_dir = "../data/supervised_list.csv"

# save dir
full_labeled_dir = "../data/wav/full_labeled/"
half_labeled_dir = "../data/wav/half_labeled/"
un_labeled_dir = "../data/wav/un_labeled/"

# wavファイルの分割
def split_wav(row, path):
    wav_data = AudioSegment.from_file(wav_dir + str(row[0]) + ".wav")

    start = float(row[2]) * 1000        # ミリ秒に変換
    end = float(row[3]) * 1000

    split = wav_data[start:end]     # wavファイルを分割
    split.export(path + str(row[0]) + "_" + str(row[1]) + ".wav", format="wav")     # 保存


def main():
    cnt_unlabeled = 0        # init var
    cnt_half_labeled = 0
    cnt_full_labeled = 0

    csv_file = pd.read_csv(csv_dir, encoding='UTF-8', header=0)

    for row in csv_file.values:
        print(row[6])       # DEBUG

        if pd.isnull(row[6]):      # ラベルなしはスキップ
            if row[5] != "{*}":    # '声喩'のみの発話はスキップ
                print("[run split_wav()] UN LABELED\n")
                split_wav(row, un_labeled_dir)
                cnt_unlabeled += 1

        elif pd.isnull(row[9]):    # 'ans_n'のみラベルあり
            print("[run split_wav()] HALF LABELED\n")
            split_wav(row, half_labeled_dir)
            cnt_half_labeled += 1

        else:                      # 'emotion'ラベルあり
            print("[run split_wav()] FULL LABELED\n")
            split_wav(row, full_labeled_dir)
            cnt_full_labeled += 1
            

if __name__ == "__main__":
    main()