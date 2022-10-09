# Created by RitsukiShuto on 2022/10/09.
# vectorization.py
# 音声、言語の生データをベクトル化する。
# 
import pandas as pd
import glob

# メタデータの読み込み
# INFO: 読み込ませるデータセットはここで変更する。
meta_data_dir = "data/OGVC_Vol1_supervised.csv"        # OGVC vol.1
#meta_data_dir = "/data/OGVC2_metadata.csv"             # OGVC vol.2

meta_data = pd.read_csv(meta_data_dir, header=0)

# 音声データの読み込み
#wav_list = glob.glob("/data/wav/OGVC_vol1/all_data/*.wav")

# 学習データとして用いる文字数の基準値
LEN = 4

def main():
    # 分類方法とメタデータでif分岐
    # ベクトル化の関数は共通
    print(meta_data.head())     # DEBUG
    docs = []

    for row in meta_data.values:
        if row[5] != "{笑}":           # 声喩のみの発話はスキップ
            if len(row[5]) > LEN:         # LEN文字以下の発話は不採用
                if pd.isnull(row[9]):       # ラベルなしデータ
                    wav_file = str(row[0]) + "_" + str(row[1]) + ".wav"
                    print("[UN LABELED]", wav_file)
                    docs.append(row[5])

                else:
                    wav_file = str(row[0]) + "_" + str(row[1]) + ".wav"
                    print("[LABELED]", wav_file)
                    docs.append(row[5])

    print(docs)     # DEBUG

if __name__ == '__main__':
    main()