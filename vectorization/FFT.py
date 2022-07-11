# Created by RitsukiShuto on 2022/06/22.
# wavファイルを取得してFFTを行う。<試作版>
#
import wave
import glob
import os

import numpy as np
import scipy as sp
from scipy.fft import fft, fftfreq
from scipy.io.wavfile import read, write

import pandas as pd
import matplotlib.pyplot as plt

# PATHを指定
data_dir = "../train_data/sound/"
save_dir = "../train_data/FFT/"

N = 5000

dir_list = glob.glob(data_dir + "*")

for dir_name in dir_list:
    file_list = glob.glob(dir_name + "/*.wav")
    # print("open", dir_name)    # DEBUG

    for file_name in file_list:
        # print("open", file_name)    # DEBUG
        fs, data = read(file_name)

        print(os.path.basename(file_name))
        # X = fft(data)
        # print(X.shape)