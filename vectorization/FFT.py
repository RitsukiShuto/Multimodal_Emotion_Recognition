# Created by RitsukiShuto on 2022/06/22.
# wavファイルを取得してFFTを行う。<試作版>
#
import glob
import pandas as pd
import wave
import numpy as np
import random

wav_dir = glob.glob("../data/wav/labeled/*.wav")
fft_list = []

# wavファイルをロード
def wave_load(filename):
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

# FFT
def fft_load(wav_file, size, start, end):
    st = 1000   # サンプリングする開始位置
    hammingWindow = np.hamming(size)    # ハミング窓
    fs = 44100 #サンプリングレート
    d = 1.0 / fs #サンプリングレートの逆数
    freqList = np.fft.fftfreq(size, d)
    
    n = random.randint(start,end)
    wave = wave_load(wav_file)
    windowedData = hammingWindow * wave[st:st+size]  # 切り出した波形データ（窓関数あり）
    data = np.fft.fft(windowedData)
    data = abs(data) ** 2

    return data

for wav_file in wav_dir:
    wave_load(wav_file)

    #print(wav_file)
    fft_data = fft_load(wav_file, 256, 0, 1000)
    fft_list.append(fft_data)

np.savetxt("../vector/FFT/FFT.csv", fft_list, fmt='%12f', delimiter=',')