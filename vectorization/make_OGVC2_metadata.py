# Created by RitsukiShuto on 2022/11/17.
# make_OGVC2_metadata
#
import glob
import os
import pandas as pd
import numpy as np

def main():
    wav_dir = 'data/wav/feature_vector/'     # wavファイルのディレクトリ
    ref_text = pd.read_csv('data/OGVC2_metadata.csv')       # テキストデータの参照元

    # OGVC Vol.1を読み込む
    ogvc_1 = pd.read_csv("data/OGVC_Vol1_supervised_5emo.csv")
    ogvc_1 = ogvc_1.drop(columns=['start', 'end', 'ans1', 'ans2', 'ans3'], axis=1)      # 不要なカラムを削除
    ogvc_1['lv'] = 9                                                                    # OGVC Vol.1に感情強度のカラムを追加
    ogvc_1 = ogvc_1.reindex(columns=['fid', 'no', 'person', 'text', 'lv', 'emotion'])   # カラムを入れ替え

    actors = ['FOY', 'FYN', 'MOY', 'MTY']                           # 演者のリスト

    # 演者ごとに処理
    for actor in actors:
        maked_ogvc2_list = []       # 演者ごとにメタデータを作成するため、ここでinit

        # OGVC2のメタデータからwavファイル名を作成する
        for row in ref_text.values:
            if row[3][:3] not in ['ANG', 'JOY', 'SAD', 'SUR', 'NEU']:
                continue

            if row[1] < 10:
                no = '0' + str(row[1])      # 'no'が1桁のときは先頭に'0'を追加する。
            else:
                no = str(row[1])            # 2桁のときはそのまま

            # 感情レベルの個数分ループ
            for lv in range(4):
                file_name = actor + row[0][3:5] + no + row[3][:3] + str(lv)
                maked_ogvc2_list.append([file_name, None, row[2], row[4], lv, row[3][:3]])

                # wavファイルが存在するかチェック
                is_file = os.path.isfile(wav_dir+file_name+'.wav')

                if is_file:
                    print(f"{wav_dir+file_name+'.wav'} is found.")

                else:
                    print(f"[ERROR!]{wav_dir+file_name+'.wav'} is not found.")

        # DataFrameに変換
        columns = ['fid', 'no', 'person', 'text', 'lv', 'emotion']      # 新しいメタデータのカラム
        df_ogvc2 = pd.DataFrame(maked_ogvc2_list, columns=columns)
        concat = pd.concat([ogvc_1, df_ogvc2])

        # 演者ごとに保存
        save_dir = 'data/' + actor + '_metadata_5emo.csv'
        concat.to_csv(save_dir, header=1, index=0)                      # type: ignore

if __name__ == '__main__':
    main() 