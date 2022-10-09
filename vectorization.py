# Created by RitsukiShuto on 2022/10/09.
# vectorization.py
# 音声、言語の生データをベクトル化する。
# 
import pandas as pd
import glob

import numpy as np

import wave

# wavファイルを読み込む
# pow_spectrum()から呼び出される
def load_wav(filename):
    # open wave file
    wf = wave.open(filename,'r')
    channels = wf.getnchannels()

    # load wave data
    chunk_size = wf.getnframes()
    amp  = (2**8) ** wf.getsampwidth() / 2
    data = wf.readframes(chunk_size)        # バイナリ読み込み
    data = np.frombuffer(data,'int16')      # intに変換
    data = data / amp                       # 振幅正規化
    data = data[::channels]
   
    return data

# FFTを行いパワースペクトルを計算する
# main()から呼び出される
def pow_spectrum(wav_file, size):
    st = 1000                       # サンプリングする開始位置
    fs = 44100                      # サンプリングレート
    d = 1.0 / fs                    # サンプリングレートの逆数

    hammingWindow = np.hamming(size)                    # ハミング窓
    freqList = np.fft.fftfreq(size, d)
    wave = load_wav(wav_file)
    windowedData = hammingWindow * wave[st:st+size]     # 切り出した波形データ（窓関数あり）
    data = np.fft.fft(windowedData)
    data = abs(data) ** 2

    return data

def main():
    # メタデータの読み込み
    # INFO: 読み込ませるデータセットはここで変更する。
    meta_data = pd.read_csv("data/OGVC_Vol1_supervised.csv", header=0)
    #meta_data = pd.read_csv("/data/OGVC2_metadata.csv", header=0)

    # 学習に用いるデータの文字数の基準値
    LEN = 4

    # wavファイルのディレクトリ
    wav_dir = "data/wav/OGVC_vol1/all_data/"

    docs = []
    labeled_pow_list = []
    unlabeled_pow_list = []

    for row in meta_data.values:
        if row[5] != "{笑}" and len(row[5]) > LEN:           # 声喩のみの発話とLEN文字以下の発話をスキップ
            if pd.isnull(row[9]):       # ラベルなしデータ
                wav_file_name = str(row[0]) + "_" + str(row[1])

                print("[INFO]", wav_file_name, "is un labeled data.")     # DEBUG
                print("[INFO]Prosessing", wav_file_name)

                # パワースペクトルを取得
                pow_spect = pow_spectrum(wav_dir + wav_file_name + ".wav", 64)
                unlabeled_pow_list.append(pow_spect)

                docs.append(row[5])                 # テキストのみ抽出

            else:                       # ラベルありデータ
                wav_file_name = str(row[0]) + "_" + str(row[1])

                print("[INFO]", wav_file_name, "is labeled data.")     # DEBUG
                print("[INFO]Prosessing", wav_file_name)

                # パワースペクトルを取得
                pow_spect = pow_spectrum(wav_dir + wav_file_name + ".wav", 64)
                labeled_pow_list.append(pow_spect)

                docs.append(row[5])                 # テキストのみ抽出


    #print(docs)     # DEBUG
    print(labeled_pow_list)

if __name__ == '__main__':
    main()