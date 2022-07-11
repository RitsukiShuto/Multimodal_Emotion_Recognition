# Created by RitsukiShuto on 2022/06/22.
# wavファイルからMFCCを求める
#
from lib2to3.pytree import convert
from audiomentations import Compose, AddGaussianNoise, Shift
import librosa as lr

import os
import random
import glob

import pandas as pd
import numpy as np

# MFCC
def MFCC(file_name):
    print('Converting ', file_name)
    mfcc = lr.feature.mfcc(y = y, sr = sr, n_mfcc = 12)
    ceps = mfcc.mean(axis = 1)

    list_MFCC.append(ceps)

# PATHを指定
data_dir = "../train_data/sound/"
save_dir = "../train_data/MFCC/MFCC.csv"

list_MFCC = []

dir_list = glob.glob(data_dir + "*")

for dir_name in dir_list:
    file_list = glob.glob(dir_name + "/*.wav")
    print("open", dir_name)    # DEBUG

    for file_name in file_list:
        print("open", file_name)    # DEBUG
        y, sr = lr.core.load(file_name, sr = None)
        MFCC(file_name)


# データフレーム化
# 20次元のMFCCデータフレームを作成
df_ceps = pd.DataFrame(list_MFCC)

columuns_name = []  # カラム名を"dict + 番号"で示す
for i in range(12):
    columuns_name_tmp = 'dict{0}'.format(i)
    columuns_name.append(columuns_name_tmp)

df_ceps.columns = columuns_name

df_ceps.to_csv(save_dir)