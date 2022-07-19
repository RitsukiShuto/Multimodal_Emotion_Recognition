# Created by RitsukiShuto on 2022/07/19.
# Power Spectrum.py
#
from json import load
import numpy as np
import wave

import pandas as pd
import glob
import os

full_labeled_dir = "../vector/FFT/full_labeled_pow.csv"
half_labeled_dir = "../vector/FFT/half_labeled_pow.csv"
un_labeled_dir = "../vector/FFT/un_labeled_pow.csv"

full_labeled_pow = []
half_labeled_pow = []
un_labeled_pow = []

def load_wav(filename):
    # open wave file
    wf = wave.open(filename,'r')
    channels = wf.getnchannels()

    # load wave data
    chunk_size = wf.getnframes()
    amp  = (2**8) ** wf.getsampwidth() / 2
    data = wf.readframes(chunk_size)   # バイナリ読み込み
    data = np.frombuffer(data,'int16') # intに変換
    data = data / amp                  # 振幅正規化
    data = data[::channels]
   
    return data

def pow_spectrum(wav_file, size):
    st = 1000   # サンプリングする開始位置
    fs = 44100 #サンプリングレート
    d = 1.0 / fs #サンプリングレートの逆数

    hammingWindow = np.hamming(size)    # ハミング窓
    freqList = np.fft.fftfreq(size, d)
    wave = load_wav(wav_file)
    windowedData = hammingWindow * wave[st:st+size]  # 切り出した波形データ（窓関数あり）
    data = np.fft.fft(windowedData)
    data = abs(data) ** 2

    return data
    

def main():
    dir_list = glob.glob("../data/wav/*")

    for dir in dir_list:
        wav_list = glob.glob(dir + "/*.wav")

        for wav in wav_list:
            if dir == "../data/wav/full_labeled":
                print("[FULL LABELED]" ,wav)
                pow = pow_spectrum(wav, 128)
                full_labeled_pow.append(pow)
            
            elif dir == "../data/wav/half_labeled":
                print("[HALF LABELED]", wav)
                pow = pow_spectrum(wav, 128)
                half_labeled_pow.append(pow)

            else:
                print("[UN LABELED]", wav)
                pow = pow_spectrum(wav, 128)
                un_labeled_pow.append(pow)

    np.savetxt("../vector/pow_spectrum/full_labeled_pow.csv", full_labeled_pow, fmt='%8f', delimiter=',')
    np.savetxt("../vector/pow_spectrum/half_labeled_pow.csv", half_labeled_pow, fmt='%8f', delimiter=',')
    np.savetxt("../vector/pow_spectrum/un_labeled_pow.csv", un_labeled_pow, fmt='%8f', delimiter=',')

if __name__ == '__main__':
    main()