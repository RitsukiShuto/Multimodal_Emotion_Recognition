# Created by RitsukiShuto on 2022/07/19.
# pow_spectrrum.py
# wavファイルを読み込みFFTを行い、パワースペクトルを計算する。
#
from json import load
import numpy as np
import wave

import pandas as pd
import glob
import os

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
    st = 1000               # サンプリングする開始位置
    fs = 44100              #サンプリングレート
    d = 1.0 / fs            #サンプリングレートの逆数

    hammingWindow = np.hamming(size)                    # ハミング窓
    freqList = np.fft.fftfreq(size, d)
    wave = load_wav(wav_file)
    windowedData = hammingWindow * wave[st:st+size]     # 切り出した波形データ（窓関数あり）
    data = np.fft.fft(windowedData)
    data = abs(data) ** 2

    return data
    

def main():
    # 出力先
    labeled_dir = "../train_data/2div/labeled_pow.csv"
    un_labeled_dir = "../train_data/2div/un_labeled_pow.csv"

    # パワースペクトル格納用変数
    labeled_pow = []
    un_labeled_pow = []

    # wavファイルのあるディレクトリ一覧を取得
    dir_list = glob.glob("../data/wav/*")

    for dir in dir_list:                        # wavファイルのあるディレクトリを1つずつ走査
        wav_list = glob.glob(dir + "/*.wav")    # wavファイルの一覧を取得

        for wav in wav_list:                    # wavファイルを1つずつ走査
            # ラベルの付与状態ごとに出力先を分ける
            if dir == "../data/wav/full_labeled":       # 'emotion'ラベル付き
                print("[FULL LABELED]" ,wav)
                pow = pow_spectrum(wav, 128)
                labeled_pow.append(pow)

            else:                                       # ラベルなし
                print("[UN LABELED]", wav)
                pow = pow_spectrum(wav, 128)
                un_labeled_pow.append(pow)

    # CSVで保存
    np.savetxt(labeled_dir, labeled_pow, fmt='%8f', delimiter=',')
    np.savetxt(un_labeled_dir, un_labeled_pow, fmt='%8f', delimiter=',')

if __name__ == '__main__':
    main()