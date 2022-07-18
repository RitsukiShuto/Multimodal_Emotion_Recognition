# Crreated by RitsukiShuto on 2022/07/18.
# STFT.py
#
import numpy as np
from scipy import signal
import math
import soundfile as sf
import glob

wav_dir = glob.glob("../data/wav/labeled/*.wav")
save_dir = "../vector/STFT/"

Zxx_list = []

for wav_list in wav_dir:
    data, sample_rate = sf.read(wav_list)
    f, t, Zxx = signal.stft(data, fs = 1024, nperseg=512)

    Zxx_list.append(Zxx)
    print(Zxx.shape)

#print(Zxx_list)