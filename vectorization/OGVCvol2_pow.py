# Created by RitsukiShuto on 2022/10/04.
# OGVC Vol2のwavファイルを感情レベルに応じて分類する
# パワースペクトルを計算する
#
import glob
import numpy as np
import wave

import pandas as pd
import os
import glob

from natsort import natsorted

data_dir = glob.glob("../data/wav/OGVC_vol2/all_data/*.wav")
save_dir = "../train_data/OGVC_vol2/"

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
    lv0_pow = []
    lv1_pow = []
    lv2_pow = []
    lv3_pow = []

    # wavファイルの処理
    for wav in data_dir:
        fname, ext = os.path.splitext(os.path.basename(wav))
        idx = str.rfind(wav, ".")
        emo_lv = idx - 1    # 感情レベルは拡張子の1文字前にある

        if wav[emo_lv] == '0':
            print("emotion level 0:", wav)       # DEBUG
            pow = pow_spectrum(wav, 64)
            data =  np.hstack((fname, pow))
            lv0_pow.append(data)

        elif wav[emo_lv] == '1':
            print("emotion level 1:", wav)
            pow = pow_spectrum(wav, 64)
            data =  np.hstack((fname, pow))
            lv1_pow.append(data)

        elif wav[emo_lv] == '2':
            print("emotion level 2:", wav)
            pow = pow_spectrum(wav, 64)
            data =  np.hstack((fname, pow))
            lv2_pow.append(data)

        elif wav[emo_lv] == '3':
            print("emotion level 3:", wav)
            pow = pow_spectrum(wav, 64)
            data =  np.hstack((fname, pow))
            lv3_pow.append(data)

        else:
            print("[ERROR]", wav)

    df0 = pd.DataFrame(lv0_pow)
    df1 = pd.DataFrame(lv1_pow)
    df2 = pd.DataFrame(lv2_pow)
    df3 = pd.DataFrame(lv3_pow)

    df0.to_csv(save_dir + "POW_lv0.csv", index=False, header=1)
    df1.to_csv(save_dir + "POW_lv1.csv", index=False, header=1)
    df2.to_csv(save_dir + "POW_lv2.csv", index=False, header=1)
    df3.to_csv(save_dir + "POW_lv3.csv", index=False, header=1)


if __name__ == "__main__":
    main()