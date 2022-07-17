# Created by RitsukiShuto on 2022/06/22.
# wavファイルを取得してFFTを行う。<試作版>
#
import glob
import pandas as pd
import wave
import numpy as np

wav_list = glob.glob("../data/wav/labeled/*.wav")

fft_list = []
for wav_file in wav_list:
    #print(wav_file)        # DEBUG

    w = wave.open(wav_file, 'rb')
    data = w.readframes(w.getnframes())
    w.close()

    fs = w.getframerate()
    s = (np.frombuffer(data, dtype="int16") / 32767.0)[0:fs]

    F = np.fft.fft(s)
    F_abs = np.abs(F)
    F_a = F_abs / fs * 2
    F_a[0] = F_abs[0] / fs

    fft_list.append(F_a)
    #print(F_a)     # DEBUG: FFT後の数値
    #print(F_a.shape)        # DEBUG: FFT後の配列サイズ

# BUG: wavファイルごとに長さが異なるため、FFTした際の配列サイズに差異が生じcsvとして保存できない
#np.savetxt("../vector/FFT/fft.csv", fft_list, fmt='%12.8f', delimiter=',')