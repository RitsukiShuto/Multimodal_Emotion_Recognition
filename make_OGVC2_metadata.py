# Created by RitsukiShuto on 2022/11/17.
# make_OGVC2_metadata
#
import glob
import os
import pandas as pd
import numpy as np

def make_ogvc2_metadata(wav_list):
    # テキストデータの参照元
    ref_text = pd.read_csv('data/OGVC_Vol2_supervised.csv')
    #ref_text = ref_text.values.tolist()

    columns = ['fid', 'no', 'person', 'text', 'lv', 'emotion']      # メタデータのカラム
    maked_ogvc2_list = []

    for row in ref_text.values:
        if row[2] not in ['ANG', 'JOY', 'NEU', 'SAD', 'SUR']:       # 5種類の感情ラベル以外はスキップ
            #print("skip")

            continue

        for lv in range(4):
            f_name, ext = os.path.splitext(os.path.basename(wav_list[lv + row[0]]))
            maked_ogvc2_list.append([f_name, None, row[1], row[3], lv, row[2]])

    

    # DataFrameにして返す
    df_ogvc2 = pd.DataFrame(maked_ogvc2_list, columns=columns)

    # OGVC Vol.1を読み込む
    ogvc_1 = pd.read_csv("data/OGVC_Vol1_supervised_5emo.csv")
    ogvc_1 = ogvc_1.drop(columns=['start', 'end', 'ans1', 'ans2', 'ans3'], axis=1)      # 不要なカラムを削除
    ogvc_1['lv'] = 9                                                                    # 感情強度のカラムを追加
    ogvc_1 = ogvc_1.reindex(columns=['fid', 'no', 'person', 'text', 'lv', 'emotion'])   # カラムを入れ替え

    concat_F = pd.concat([ogvc_1, df_ogvc2])

    return concat_F

def main():
    # 処理するwaxファイルのリストを取得
    wav_dir = "data/wav/mixed/"                     # wavファイルのディレクトリ
    wav_list_F = glob.glob(wav_dir+"FOY*.wav") + glob.glob(wav_dir+"FYN*.wav")        # 女性の演技音声のリストを取得
    wav_list_M = glob.glob(wav_dir+"MOY*.wav") + glob.glob(wav_dir+"MTY*.wav")        # 男性の演技音声のリストを取得

    print(len(wav_list_F))
    print(len(wav_list_M))

    df = make_ogvc2_metadata(wav_list_F)
    df.to_csv('data/OGVC_mixed_F.csv', header=1, index=0)

    df = make_ogvc2_metadata(wav_list_M)
    df.to_csv('data/OGVC_mixed_M.csv', header=1, index=0)


if __name__ == '__main__':
    main()